+++
title = "The Transmeta Code Morphing Software"

[[extra.authors]]
name = "Ryan Doenges"
link = "http://ryandoeng.es/"

extra.bio = """
  [Ryan Doenges](http://ryandoeng.es/) is a third year PhD student studying
  programming languages and networking.
"""
+++

Today we are reading the 2003 paper on Transmeta CMS (Code Morphing
Software<sup>TM</sup>). The CMS layer ran x86 programs on the Transmeta
Corporation's Crusoe microprocessor, which had an internal architecture that was
much simpler than an x86. While much of the terminology in the paper is
non-standard, I hope it was clear that CMS is a just-in-time (JIT) compiler for
x86 targeting Crusoe's internal instruction set architecture (ISA).

## Why a JIT?
How and when the computer industry settled on (for?) the x86 instruction set
architecture I do not know, but we can surmise from the engineering effort
expended by Transmeta that it happened before 2003. At the time, new general
purpose processors needed to expose an x86 interface to programmers.

Faced with this task, Transmeta engineers could have gone the obvious route and
built an x86 clone in hardware. The paper argues for the internal ISA and CMS
technique as follows.

> This approach allows a simple, compact, low-power microprocessor
> implementation, with the freedom to modify the internal ISA between
> generations, while supporting the broad range of legacy x86 software
> available.

To flip all these adjectives, the Crusoe designers recognized that a direct
x86 implementation like Intel's was complicated, sprawling, and high-power. They
understood that Intel's infamous commitment to backwards compatibility was good
for business but made it difficult to modify hardware when it might improve
maintainability, space usage, or power efficiency. Hiding their actual
architecture behind an x86 abstraction solved this problem.

## The internal ISA

Crusoe's internal ISA is a VLIW (very long instruction word) ISA with 64
general-purpose registers and 32 floating-point registers, which is more than
the x86. In a VLIW instruction set like Crusoe's, each instruction is really
several smaller instructions which are issued in parallel. In the terminology of
the paper the large instructions are "molecules" composed of 2 or 4 "atoms".
The internal ISA avoids handling pipeline stalls---instead it expects the CMS
compiler to generate safe code by separating conflicting operations.

The hardware supports deoptimization by shadowing state and exposing `commit`
and `rollback` operations for copying live state to the shadowed state and
reverting to shadowed state respectively. In particular, every register has
a corresponding shadow register. All writes to memory are held in a gated store
buffer that is only flushed to main memory following a `commit`.

## The Code Morphing System
The CMS includes a software x86 interpreter which runs programs accurately while
also monitoring performance statistics. Once it notices a particular code region
has run more than some threshold number of times, it stops interpreting, commits
the current state, and tries running a just-in-time compiled version of the code
region. The compiled code is stored in a "translation cache" or Tcache.

The JIT will reorder instructions in order to get an efficient schedule on the
VLIW architecture. This is necessary for performance: Figure 2 in the paper
shows a mean of 33% performance degradation across several applications when
reordering is disabled.

### Exceptions and Interrupts
Occasionally compiled code will encounter exceptions in the internal ISA.
Sometimes these exceptions are the result of speculative compilation (e.g.,
a reordering of instructions causing a memory fault) but sometimes they are
genuine exceptions which should be propagated up to the x86 layer (e.g.,
division by zero).

When Tcached code hits an exception, the CMS issues a rollback instruction to
restore architectural state to a previous checkpoint and tries interpreting the
region instead. If the exception goes away, CMS assumes it was due to
reordering. Otherwise it is a genuine x86 exception and gets propogated up to
the program.

Interrupts work similarly to exceptions, but CMS does not try retranslating the
region in which the interrupt occurs.

### Reordering constraints
Consider a program that writes a 1 to address `x` and then reads from address
`y`. If these two pointers alias, it is unsafe to reorder the read to run before
the write. Similarly, if `x` and `y` are backed by a memory-mapped I/O device,
reordering the operations would be unsafe because it would cause the program's
I/O behavior to change.

The CMS speculatively optimizes with reordering and handles these potential
issues by turning them into faults, which trigger deoptimization before anything
bad can happen. Reorderd instructions are tagged to let the processor know they
were reordered. Special "alias hardware" does lightweight alias tracking at run
time and faults if there may be aliasing between two reordered operations.
Reordered operations that access IO address space also fault. The offending code
region is then recompiled without the reorderings.

### Self-modifying code
It is not uncommon for x86 programs to modify themselves. The paper observes
that it is a standard technique in games, embedded code, and Windows device drivers.
Following the approach of turning correctness issues into faults, Transmeta
could (and apparently at one point did) write-protect code pages to cause faults
and then fall back to interpreting the self-modifying code. Falling back to
the interpreter is a serious performance penalty for self-modifying programs, so
the paper includes a few techniques for handling self-modifying code.

Finer-grained write protection can help, since code is likely to be modified
only in a few places. Crusoe supports this and it gets some speedup over the
page granularity write protection approach.

Introducing "prologues" that check preconditions on translated code can also
work. The paper refers to this as self-validation and self-checking. The idea is
to tack a header onto the translated code which looks up the source page and
verifies that nothing has changed since it was translated. 

Finally, it is possible to recognize common self-modifying code patterns and
compile them to ordinary static code. 

# Conclusion
The Transmeta CMS system is a compiler solution to a hardware problem. While
implemnting a just-in-time compiler for x86 is subtle and difficult, the
performance and maintainability benefits for user code and the hardware seem
worth it.
