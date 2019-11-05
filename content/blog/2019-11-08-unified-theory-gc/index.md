+++
title = "A Unified Theory of Garbage Collection"
extra.author = "Mark Anastos"
extra.author_link = "https://github.com/anastos"
extra.bio = """
  Mark Anastos is an undergraduate senior studying Computer Science and
  Electrical & Computer Engineering.
"""
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

## Intuition 

## Tracing & Reference Counting are Duals

On the high level, tracing is tracking "matter" -- all reachable objects, while reference counting is tracking "anti-matter" -- all unreachable objects. Their connection is further revealed when we align them by removing certain "optimizations". We can consider a version of tracing that computes the number of incoming edges from live objects instead of a single bit; and a version of reference counting that postpones the decrements to be processed on batches. If the graph contains no cycle, both methods would converge to tagging the same value for each object. Tracing achieves this by setting this value to zero and increases it recursively, while reference counting starts from an upper bound and decrements it recursively. 

To formalize this connection, we define the value they converge to mathematically then align their algorithmic structures.

## Mathematical Model

### Mathematical Model

fixed-point formulation
minimal vs maximal fixed point -- cycles
strategies for collecting cycles
    backup tracing
    trial deletion

## Tracing & Reference Counting are Duals

### Alignments of Algorithmic Strctures
explain algorithms and how they are related/opposites

## Hybrids

### Deferred Reference Counting & Partial Tracing

### Generational Collectors

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
