+++
title = "A Unified Theory of Garbage Collection"
extra.author = "Mark Anastos and Qian Huang"
extra.author_link = "https://github.com/anastos"
extra.bio = """
  Mark Anastos is an undergraduate senior studying Computer Science and
  Electrical & Computer Engineering. Qian Huang is an undergraduate junior studying Computer Science and Mathematics.
"""
extra.latex = true
+++

## Introduction

Tracing and reference counting are normally being viewed as completely different approaches to garbage collection. However, in A Unified Theory of Garbage Collection, David et al. show that they are in fact duals of each other through a particular formulation. Intuitively, tracing is tracking the live objects while reference counting is tracking dead objects. They further showed that all high-performance collectors are in fact hybrids of tracing and reference counting.


## Background

Broadly speaking, garbage collection (GC) is a form of automatic memory management. The garbage collector attempts to free the memory blocks occupied by objects that are no longer in use by the program. It relieves programmers from the burden of explicitly freeing allocated memory. Moreover, it also serves as part of the security strategy of languages like Java: in Java virtual machine programmers are unable to accidentally (or purposely) crash the machine by incorrectly freeing memory. The opposite is manual memory management, which is available in C/C++. This gives the maximum freedom for programmers and avoids the potential overhead that affects program performance.

The task garbage collection needs to solve is identifying the objects not accessible by the program in the reference graph. It then frees the unreachable objects and rearranges the memory sometimes to reduce heap fragmentation. 

The most traditional approaches are tracing and reference counting:
- Trancing
  Recursively mark reachability by starting from a set of roots memory blocks that are in use (e.g. pointed by global variable or local variable currently in stack frames).

- Reference Counting 
  Count the number of pointers pointed to one particular object by bookkeeping it every time a pointer is created or modified. It frees the object when the counter decreases to zero.

These two approaches have a lot differences:

<img src="diff.png" style="width: 100%">

Although tracing naturally solves the reachability problem accurately, it requires to traverse over a static graph and therefore suspend the whole program. On the other hand, reference counting is done incrementally along with each pointer assignment and collect. However, it brings unnecessary overhead when the pointers are changed often and it does not collect cycles of garbages. Thus people proposed more complicated algorithms based on different hypotheses, such as deferred reference counting, generational garbage collection, etc. 

## Tracing & Reference Counting are Duals

On the high level, tracing is tracking "matter" -- all reachable objects, while reference counting is tracking "anti-matter" -- all unreachable objects. Their connection is further revealed when we align them by removing certain "optimizations". We can consider a version of tracing that computes the number of incoming edges from roots or live objects instead of a single bit; and a version of reference counting that postpones the decrements to be processed on batches. If the graph contains no cycle, both methods would converge to tagging the same value for each object. Tracing achieves this by setting this value to zero and increases it recursively, while reference counting starts from an upper bound and decrements it recursively. 

To formalize this connection, we define the value they converge to mathematically then align their algorithmic structures.

### Mathematical Model

In order to analyse and compare different garbage collection strategies, the
paper presents a mathematical model of the memory management problem that
garbage collection is trying to solve. Objects in memory and the pointers that
they contain are modeled as the nodes and edges of a graph. The set of objects
is denoted as $V$, and the multiset of pointers between objects is $E$. An
object should not be freed if it could be used in the future. A conservative
approximation of this, without any program analysis, is that an object could be
used in the future if there exists a path of pointers to the object which
originates from the stack or from a register. We call the starting points of
such paths (i.e., all objects to which there is a direct pointer on the stack or
in a register) the roots of the graph, which make up the multiset $R$.

Using these definitions, we can formulate the reference counts of objects
(denoted $\rho(v)$ for $v \in V$) as a fixed point of the following equation:

$$ \rho(v) = \big|[v : v \in R]\big| +
             \big|[(w, v) : (w, v) \in E \land \rho(w) > 0]\big| $$

Here we recursively define the reference count of an object $x$ to be the number
of root pointers to $x$ plus the number of pointers to $x$ from objects which
themselves have non-zero reference counts. Any object whose reference count is
zero according a fixed point of $\rho$ can be freed, as there is no way for the
program to reference it in the future.

However, it is important to note that there could be multiple fixed points to
this equation, namely in the presence of cyclic garbage. If object $a$ points to
object $b$, and vice versa, but neither is a root and neither is pointed to from
elsewhere, then, by this formulation, both $\rho(a) = \rho(b) = 1$ and $\rho(a)
= \rho(b) = 0$ are valid solutions. Ideally, a garbage collection algorithm will
find the least fixed point of $\rho$, meaning that it will consider all cyclic
garbage as able to be freed. A tracing collector does this, whereas a reference
counting collector does not detect any cyclic garbage, and thus finds the
greatest fixed point.

strategies for collecting cycles
    backup tracing
    trial deletion

### Alignments of Algorithmic Strctures
explain algorithms and how they are related/opposites


comment about using special properties?


## Hybrids

The authors further show that all realistic garbage collectors are in fact hybrids of tracing and reference counting. In general we can categorize collectors to unified heap collectors, split heap collectors and multi-heap collectors. Then different garbage collectors can be seen as performing tracing or reference counting when tracking references within each region and cross regions.

### Unified Heap collectors: Deferred Reference Counting & Partial Tracing

<img src="deferred.png" alt="Snow" style="width:100%"> |  <img src="partial.png" alt="Snow" style="width:100%">
:-------------------------:|:-------------------------:
Deferred Reference Counting |  Partial Tracing

Rather than doing reference counting completely, Deferred Reference Counting defers updating the reference count of objects pointed directly by roots until batch processing. This is based on the observation that pointers from roots are likely to change very often as they are directly used in the program. Notice that we can view this as tracing from roots to their targets and reference counting for the intra-heap pointers: All the assignments that lead to intra-heap pointers change would be tracked by reference counting as normal. When we suspend the program, we trace the roots for one level, which compensates the delay.

Reversely, we could design Partial Tracing, which uses reference counting for edges from roots to heaps while tracing the intr-heap pointers. However, this combines the worse properties of both tracing and reference counting: it suffers from the high mutation cost from the fast changing of root pointers while still need to spend long time to trace the heap. This design failure demonstrates that although tracing and reference counting are duals, they are not equally easy to solve under different cases. 

### Split Heap Collectors: Generational Collectors

Generational collectors are based on the following emperical observation that most objects are short lived, as shown in the left figure bellow. Here Y axis shows the number of bytes allocated and the X access shows the number of bytes allocated over time. (https://www.oracle.com/webfolder/technetwork/tutorials/obe/java/gc01/index.html)

<img src="ObjectLifetime.gif" alt="" style="width:100%"> | <img src="gen.png" alt="" style="width:100%"> 
:-------------------------:|:-------------------------:
Most objects are short lived  |  Generational Collectors


So Generational Collectors isolated out a nursery space from the remaining mature space. Most of the time it only collects garbage from this nursery space and moves the remaining alive objects to mature space (minor collections). Once in a while it performs a garbage collection cross the whole heap to clean the mature space (major collections). An example of Generational Collectors is available in https://blogs.msdn.microsoft.com/abhinaba/2009/03/02/back-to-basics-generational-garbage-collection/

This process can also be seen as a combination of tracing and reference counting as shown by (a) in the right figure: Reference counting is performed to track edges from mature space to nursery space. Tracing is performed within nursery space during minor collections. And finally a full tracing is performed during major collections.

We can then explore different combinations of tracing and reference counting within each space. However, notice that reference counting is always used for the edges from mature to nursery space in order to avoid tracing the whole mature space. In fact, the authors claim that any algorithm
that collects some subset of objects independently is fundamentally
making use of reference counting.

### Multi-Heap Collectors?

## Cost Analysis

define variables
general time and space formulas

## Conclusion

gc design strategies
    partitioning memory
    traversal
    trade-offs

Note that in this paper, the authors are mainly concerned with identifying unreachable objects correctly with high performance in terms of speed and space usage, probably because rearranging heap can also be done with memory allocation.
