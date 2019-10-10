+++
title = "POSH: A TLS Compiler that Exploits Program Structure"
extra.author = "Josh Acay & Drew Zagieboylo"
extra.bio = """
  [Josh Acay](https://www.coskuacay.com/) is ...
  [Drew Zagieboylo](https://www.cs.cornell.edu/~dzag/) is a 3rd year PhD student researching Security, HW design and Programming Languages. He enjoys rock climbing and gaming in his free time.
"""
+++

The development of multicore processor architectures in the 2000's led to
significant advancements in the performance of parallel computing. As a
software developer, if you could split your program or your data into
discrete chunks, you could send different pieces off to different cores
and have all of the processing done in parallel.

Naturally, software developers, compiler writers, and hardware architects
all began to wonder: _"Can we somehow use these extra cores to speed
up sequential, non-parallelizable workloads?"_
One proposed technique to answer this question is "Thread-Level Speculation" (TLS).
TLS allows software to run portions of a sequential program in parallel while
retaining the original sequential semantics. The key idea is that special hardware
support will detect when any of these parallel tasks have misbehaved; in that case,
the hardware will rollback or otherwise hide the behavior of such "speculative tasks."

In general, choosing where to insert these tasks so that they are likely to
succeed and actually provide speedup over serial execution is a difficult problem.
POSH is a compiler that automatically identifies some of these regions for you,
by using simple heuristics and profiling to eliminate candidate tasks that are
unlikely to be worth the cost of inserting them.


# TLS and Hardware Transactional Memory

Before we dive into POSH itself, we want to give a more detailed
background on both how TLS works and the context in which it was envisioned.
As we mentioned above, TLS relies on special hardware support for detecting
data dependencies between threads running on different processor cores.
Broadly, these kinds of features are known as Hardware Transactional Memory (HTM).
At the [end of this article](#hardware-transactional-memory-aside) we've included a brief aside on HTM and its
presence in modern processors for those who are interested.

POSH assumes that hardware has support
for the following features:
 - Inputs to tasks are passed via memory, not registers
 - Hardware automatically detects conflicting memory reads/writes
   between the main thread and speculative tasks
   and then automatically kills or restarts tasks
 - The ISA extension has the `spawn` and `commit` primitives
   for starting and ending task execution.

Most papers exploiting HTM rely on a very similar set of assumptions
and indeed, real HTM extensions have guarantees not unlike those listed here.
The primary difference between these assumptions and reality are empirical limitations
on code and working set size for speculative tasks.

The goal for TLS (remember, HTM is the set of hardware features, while TLS
is a software-level technique that utilizes those features),
is to speculatively parallelize code by predicting which regions do
not have real data dependencies. Existing compiler optimizations already
attempt to identify such dependencies (e.g., [instruction scheduling](../instruction-scheduling))
in order to improve performance. However, those optimizations must be
conservative in order to preserve program semantics.
Since TLS compilers can rely on runtime support from the hardware to preserve
correctness, they can aggressively overestimate data independence to
maximize potential parallelism.

# Posh Optimizations

Parallelize subroutines & loop iterations

# Sources of Performance Improvements of TLS

Parallel computation; Data pre-fecthing
(Mention using value speculation to generate more parallel computation)

# Posh Profiler & Heuristics

How to remove tasks that are likely to not help and just create overhead

# Evalutaion


### Hardware Transactional Memory Aside

Before transactional memory, hardware support for parallel computing
was limited to synchronization primitives such as [`atomic compare-and-swap`](https://en.wikipedia.org/wiki/Compare-and-swap)
or [`store-conditional`](https://en.wikipedia.org/wiki/Load-link/store-conditional).
Transactional Memory was meant to accelerate the common use case for such
primitives: atomic software transactions, consisting of a potentially unbounded number of instructions.

In this ideal world, programmers could write systems code like:
```C
withdraw(bank_acct *acct, int amt) {
  atomic {
    if (acct->balance >= amt) {
        acct->balance -= amt; return true;
    } else { return false; }
  }
}
```
where `atomic` was a hardware-supported feature for ensuring the atomicity of the contained code.
If any other thread modified `acct->balance` during the execution of this transaction,
it would *abort* and have to be retried or cancelled.

Usually, HTM is implemented by piggy-backing off of the cache coherence protocol,
which normally ensures that memory writes to the same address are eventually propagated
between cores. Unfortunately, cache coherency can be [notoriously complex](https://doi.org/10.1109/2.55497),
especially in the face of ambiguously defined and/or weak memory models.
One might reasonably expect adding new synchronization features to introduce bugs
and/or interact unexpected with existing weak memory guarantees.
Furthermore, relying on cache coherency drastically limits size of datasets read or written by hardware transactions;
in [most systems](https://researcher.watson.ibm.com/researcher/files/us-rodaira/ISCA2015_ComparisonOfHTM.pdf) the write set must fit entirely inside the L1 cache.


### HTM Today

In reality, hardware transactional memory has primarily been a failure
and does not see wide use today.
While Intel theoretically supports these kinds of instructions
with [TSX](https://en.wikipedia.org/wiki/Transactional_Synchronization_Extensions),
numerous bug reports have caused them to [disable it on a number of processors](https://www.anandtech.com/show/8376/intel-disables-tsx-instructions-erratum-found-in-haswell-haswelleep-broadwelly).
Furthermore, the [limitations of TSX](https://blog.ret2.io/2019/06/26/attacking-intel-tsx/)
and other such extensions often make using them impractical, unstable and/or insecure.

