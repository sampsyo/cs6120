+++
Title = "Evaluating the Performance Implications of Physical Addressing"
extra.author = "Drew Zagieboylo"
extra.bio = """
  [Drew Zagieboylo](https://www.cs.cornell.edu/~dzag/) is a 3rd year PhD student researching Security, Hardware Design, and Programming Languages. He enjoys rock climbing and gaming in his free time.
"""
+++


# Intro to Virtual Addressing

Modern processors use [_virtual addressing_](https://en.wikipedia.org/wiki/Virtual_address_space)
to access _actual_ memory locations through a translation layer.
Only highly privileged software, such as the operating system (OS),
has access to physical memory addresses while all other processes
can only refer to memory via these virtual addresses.
When a process requests memory (e.g., via `malloc`),
the OS will allocate physical memory in fixed size chunks, called pages,
and then map them into the process' virtual address space.
This allows the OS to allocate whichever regions of physical memory happen to be free despite the fact that the process may have requested a large, contiguous allocation.

Virtual addressing provides a few key abstractions for user-level software:
 1. A fully contiguous address space.
 2. A unique address space not shared by any other process.

The former enables software to easily calculate relative memory addresses;
accessing any element in an array requires only one or two instructions
to add the offset of the base pointer and then load from memory.
Similarly, locations on the program stack are computed relative to the current stack pointer.
Neither of these "pointer arithmetic" operations would be valid if
executed on the physical addresses.
The latter is a useful security primitive that enables
strong process memory isolation "for free," since there is no way for a process
to even reference memory owned by another process
(unless the OS maps some physical location into both address spaces).


## The Case Against Virtual Addressing

The translation of virtual addresses is accelerated by dedicated hardware
called the [Translation Lookaside Buffer](https://en.wikipedia.org/wiki/Translation_lookaside_buffer)(TLB).
This acts as a "translation cache" and hides most of the cost of virtual address translation,
except for when an address is not present in the TLB.
Missing in the TLB triggers a complex series of physical memory accesses
called "walking the page table" and tends to be extremely expensive
(especially if this has to be handled by software).

For workloads that allocate very large amounts of memory, the TLB can't actually "reach"
all of the necessary memory addresses, causing frequent
[TLB misses](https://research.cs.wisc.edu/multifacet/papers/isca13_direct_segment.pdf).
In these cases, it's not uncommon for the CPU to be running only a single application
which would like to manage its own memory anyway; the aforementioned advantages
of virtual addressing are significantly reduced but the cost in TLB misses can be devastating to performance.
The other major cause of TLB misses is frequent context switching between processes,
which typically triggers a complete flush of the TLB state. For multithreaded applications
which rely heavily on system calls (e.g., webservers), this can incur [overheads around 20%!]
(https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/osr2007_rethinkingsoftwarestack.pdf).

Furthermore, virtual addressing is not a requirement for memory security.
There are many different proposals (and even some usable implementations)
of _tagged memory_ architectures, where physical memory locations are associated with
_tags_ that control how those locations can be accessed by software.
Some examples included the [CHERI capability architecture](https://www.cl.cam.ac.uk/research/security/ctsrd/cheri/),
the [PUMP processor for software-defined metadata](https://dl.acm.org/citation.cfm?id=2694383)
and the [secure information flow CPU, Hyperflow](https://dl.acm.org/citation.cfm?id=3243743).
Instead of relying on a process' inability to address memory,
these designs use hardware to efficiently check whether or not a memory access
is allowed by the system's security policy. In these designs,
the protection provided by virtual addressing is either mostly or completely redundant.

# Removing Virtual Addressing

Let us imagine that we are running code on one of these tagged memory architectures
and we want to eliminate virtual addressing and the overheads it entails.
In this world, we can still ask our OS for memory via `malloc`; however it returns
back to us a physically contiguous memory region (rather than virtually contiguous).
For the large memory applications described above that manage their own memory,
they would likely start by `malloc`-ing most of the computer's physical memory
and then never calling `malloc` again. Little would change for such programs
(except that the spatial locality assumptions their designers had originally
made about memory layout are more likely to reflect reality).

However, programs which request new allocations throughout their lifetimes
may no longer function as often. Since `malloc` is returning a physical memory region,
the OS needs to find a large enough space inside the memory to allocate. Due to the
presence of [fragmentation](https://en.wikipedia.org/wiki/Fragmentation_(computing)),
it is possible that no such region exists. In that case, `malloc` returns `0` and,
in all likelihood, the program explodes.

Remember that such fragmentation was present
with virtual addressing as well, but the OS could stitch together various fragmented segments
to form a single virtual allocation. Therefore, programs should strive to allocate memory in fixed-size chunks;
essentially, they should assume that the OS can only allocate them pages of physical memory
and it's _their job to stripe datastructures across them_.


# Dealing With The Stack

The [stack](https://en.wikipedia.org/wiki/Call_stack) presents another potential issue.
Current compilers assume that stack-allocated variabled can be addressed relative to the
stack pointer, which is stored in a register.
Obviously, while this is an efficient mechanism for address computation,
this scheme doesn't work if any given stack frame is not comprised of physically contiguous memory.

For certain applications, it is likely that we can allocate a single stack page at startup
and then go on with our lives. In this case, the restrictions mentioned above aren't really
an issue. However, programs _may_ allocate large data structures on the stack, may recurse
deeply or may have dynamically sized stack allocations. In these cases, we can run into
the issues described above since the stack we've already allocated may not be large enough.

One solution to this problem is to dynamically allocate stack frames whenever a function call
is made. In this case, every function prologue needs to check the current stack and see if
there's enough space. If there is, then the function executes normally; otherwise,
the function asks the OS for a memory region big enough to store the current function's
entire stack frame before running. During the function epilogue, the program should
then `free` that memory.

It turns out that [gcc](https://gcc.gnu.org/) has implemented
exactly this functionality and calls it ["stack splitting"](https://gcc.gnu.org/wiki/SplitStacks).
You can check out that link for a detailed explanation of the `-fsplit-stack` option for
gcc, but it essentially implements the algorithm described above, modulo some tricks
for making the common case fast and maintaining its own small free list for stack pages.

## Overhead of Stack Checking

We evaluated the performance impact of using split stacks on several [PARSEC](https://parsec.cs.princeton.edu/)
benchmarks, as well as a microbenchmark designed to be bottlenecked on function calls.
Our microbenchmark was a naive program to compute the 50th
number in the fibonnacci sequence without memoization.
This program recurses quite a lot and thus should magnify the overhead
of checking whether or not allocation is required for each function call.




## Overhead of Dynamic Allocations

## Discussion

# Dealing With Large Objects

#Evaluation Methodology