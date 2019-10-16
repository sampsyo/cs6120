+++
title = "Threads Cannot Be Implemented as Libraries"
extra.author = "Neil Adit", "Edwin Peguero"
+++

Thread support has been appended onto languages that lack formal thread semantics as a library. 
Such libraries informally define its thread semantics, relegating precise details to the compiler implementation.

Lacking a formal concurrency model, such compilers are unable to precisely reason about the correctness of a transformation with respect to concurrency. 
More specifically, concurrency optimizations that preserve sequential correctness may break concurrent correctness, introducing undefined behavior into a program that is well-defined with respect to the thread library specification.
It is in this sense that "threads cannot be implemented as libraries"; rather, they must be implemented in the language specification.

This paper examines the case of the widely used C/C++ Pthreads library, demonstrating three kinds of compiler optimizations that can break valid Pthreads programs.

Afterwards, the authors argue for a relaxed concurrency model that eschews a race-free, critical-section based thread-programming paradigm in favor of one that sanctions data races and relies on atomic operations. 
In particular, they demonstrate performance improvements due to this paradigm shift in an implementation of the Sieve of Eratosthenes and in a garbage collector.

Finally, the authors comment on their efforts towards adding a formal thread model to the C++ standard based on the Java Memory Model.

## Pthreads: Threads Implemented as a Library

The Pthreads standard informally specifies the semantics for concurrency as follows:

> "Applications shall ensure that access to any **memory location** by more than one thread of control (threads or processes) is restricted such that ***no thread of control can read or modify a memory location while another thread of control may be modifying it***. 
> Such access is restricted using functions that synchronize thread execution and also synchronize memory with respect to other threads. The following functions synchronize memory with respect to other threads:
>
> ..., 
>
> pthread mutex lock(), 
>
> ...,
>
> ..., 
> pthread mutex unlock(), ...
>
> [Many other synchronization functions listed]"


Thus, a Pthreads program is well-defined if it lacks **data races**.
This might seem precise at first glance, but *how can we determine whether a program has a race?*
We require a semantics for threaded programs in order to evaluate whether an execution trace contains a data race, but this semantics is itself given in terms of a data race! Thus, Pthreads provides a *circular definition* for thread semantics.

Conceptually, this circularity is resolved by an implementation-defined thread semantics.
Intuitively, we may expect an implementation akin to the **sequential consistency** (SC) model, which interprets a threaded program as an interleaving of instructions across threads, such that intra-thread instruction order is preserved.

For example, under the SC model, we would observe a data race whereby at least one of `r1 == 1` or `r2 == 1` holds after the execution of the following threaded program:


```
int r1 = 0;
int r2 = 0;

// Thread 1
x = 1; r1 = y;

// Thread 2
y = 1; r1 = x;
```


However, most implementations specify a weaker model that allows for `r1 == r2 == 0` by allowing for intra-thread instruction re-ordering.

Realistically, a weaker model than SC is necessary for two reasons: 
- hardware may reorder memory operations in such a way that contradicts the SC model.
- thread-oblivious compiler optimizations are only constrained by sequential correctness, not by SC. Thus, it is legal for memory operations reordering to break SC.

## Undefined Behavior in Pthreads
This weaker model is *formally undefined* in Pthreads. This means that the behavior of a Pthreads program with a data race under the SC model can be anything: it's up to the compiler to decide. On the other hand, the behavior of a program without data races under the SC model is itself SC.

The reason behind this design decision is explained as follows:
>Formal definitions of the memory model were *rejected as unreadable by the vast majority of programmers*. 
>In addition, most of the formal work in *the literature has concentrated on the memory as provided by the hardware as opposed to the application programmer* through the compiler and runtime system. 
>It was believed that a simple statement intuitive to most programmers would be most effective

Thus, the example above exhibits undefined behavior, and so any compiler-chosen behavior is legal, including one for which `r1 == r2 == 0` holds.


## Achieving Race-Freedom Through Synchronization
To facilitate the writing of race-free, well-defined Pthreads programs, Pthread offers synchronization primitives such as the **memory barrier** and the **mutex**. 
This allows for the containment of shared memory operations in programmer-defined critical sections.
However, C++ is thread-oblivous and is therefore unaware of synchronization primitives.
Thus, it is imperative that Pthread compilers disallow all optimizations that don't preseve synchronization.
To this aim, most compilers restrict memory reorderings around synchronization primitives. 
To achieve this behavior, Pthread synchronization calls such as `pthread_mutex_lock()` are treated as **opaque functions**: functions with hidden implementations that are assumed to potentially modify all shared, global variables. 
Since the compiler cannot soundly move memory operations across opaque function calls, this implementation closely approximates the desired behavior.

Unfortunately, this implementation is unsound: it doesn't preclude other kinds optimizations from introducing data races into race-free programs, thereby rendering their behavior undefined with respect to Pthreads.

## Three Unsafe Optimizations
