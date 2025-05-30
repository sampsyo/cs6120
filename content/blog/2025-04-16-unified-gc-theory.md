+++
title = "Exploring the Merits of the Garbage Collection Spectrum"
[extra]
bio = """
  This week, we read [A Unified Theory of Garbage Collection](https://dl.acm.org/doi/10.1145/1028976.1028982) by Bacon et al. from OOPSLA 2004.
  To what extent is its conclusion, specifically that garbage collection algorithms lies on a spectrum between reference counting and tracing, 
  practical, useful, or even true?
"""
latex = true
[[extra.authors]]
name = "Ethan Gabizon"
[[extra.authors]]
name = "Ernest Ng"
[[extra.authors]]
name = "Parth Sarkar"
+++

# Background

Garbage collection, a form of automatic memory management, frees programmers from needing to deallocate memory themselves. 
The two main garbage collection algorithms are tracing (also known as mark-and-sweep) and reference counting (RC). They each present distinct performance tradeoffs -- a tracing collector (e.g. the [Java Virtual Machine](https://stackoverflow.com/questions/65312024/why-are-jvm-garbage-collectors-trace-based)'s collector) will pause execution of the program to scan the entire heap at once, whereas a reference counting collector (e.g. [CPython](https://github.com/python/cpython/blob/main/InternalDocs/garbage_collector.md)'s collector) incurs shorter pause times by incrementally tracking objects. However, reference counting collectors require some extra machinery for tracking cycles. 

The paper's central thesis is that tracing and RC, although traditionally viewed as being entirely distinct, are actually algorithmic duals. Moreover, the authors demonstrate how various (more sophisticated) GC strategies, such as generational garbage collection, can be viewed as hybrids of tracing and RC. The authors argue that this notion of duality allows one to "systematically" explore the 
design space for GCs and better select the best GC strategy for a particular application. 

# Contributions

## Qualitative analysis
By presenting tracing and RC as duals, the authors introduce a novel mental model for approaching garbage collectors. 
Specifically, tracing operates on live objects (“matter”), while RC operates on dead objects (“anti-matter”). Concretely, tracing initializes reference counts to 0 (an underestimate of the true count), *incrementing* them during graph traversal until they reach the true count. On the other hand, RC initializes reference counts to an overestimate of the true count, *decrementing* them during graph traversal until they reach the true count (ignoring cycles). (With RC, we start with an overestimate since we count in-edges from objects that are no longer live.) 

In addition, the authors formulate garbage collection as a fix-point problem, and they demonstrate that 
tracing computes the *least* fix point, while RC computes the *greatest* fix point, with their set difference being cyclic garbage.

The authors also identify several key characteristics of each algorithm:
| | Tracing | RC |
| --- | --- | --- |
| Starting point | Roots | Anti-roots |
| Graph traversal | Forward from roots | Forward from anti-roots | 
| Objects traversed | Live | Dead |

With this in mind, the authors leave the reader with 3 considerations to keep in mind when designing a new GC algorithm:
1. **Partition**: Should memory be divided into different regions which are each subjected to (possibly different) strategies?
2. **Traversal**: For each partition, should tracing or RC be used?
3. **Trade-offs**: For each partition, decide how to handle space-time trade-offs

## Hybrid collectors
The authors demonstrate that in practice, various GC strategies lie on a continuum between tracing and RC. For example, 

For collectors where there is a *unified heap* (i.e. a single heap in which all data resides), we have:
- **Deferred RC (DRC)**: References from the stack to the heap are traced, while references within the heap are reference-counted
- **Partial tracing**: Converse of DRC, i.e. reference-count roots and trace the heap

For generational GCs, where the heap is split into a nursery and a mature space, we have variations where:
- both the nursery and mature space are traced (standard generational GC)
- the nursery is RCed and the mature space is traced 
- the nursery is traced and the mature space is reference-counted (Ulterior Reference Counting)

## Cost model
The authors also present a formal cost model for comparing the performance characteristics of different collectors. Specifically, they introduce:
- $\kappa$, the time overhead for a single garbage collection
- $\sigma$, the space overhead for a single garbage collection
- $\phi$, the frequency of collection
- $\mu$, the mutation overhead
- $\tau$, the total time overhead for an entire program

Notably, the authors mention that these quantities represent "real costs with implied coefficients", as opposed to 
idealized big-Oh notation. Crucially, the authors claim that their cost model accounts for space-time tradeoffs and allows for an somewhat realistic, "apples-to-apples" comparison of different collectors' performance. 

# Merits & Shortcomings
Many people in class found the formulation of tracing and RC as algorithmic duals to be particularly elegant.
Additionally, we found their classification of different GC strategies based on how they use tracing/RC on different regions to be a 
succinct way to highlight the conceptual differences between different non-trivial GC regimes. 

Part of our discussion focused on two aspects of the authors' cost model: (1) its accuracy, and (2) its utility. Regarding (1), we decided we would have liked to see a quantitative analysis of the cost model with actual benchmarks. While it seems like an elegant abstraction, we can't know for sure if the introduced model accounts for all variations between collectors, or at least the important ones. Additionally, we were interested about the impact of the model's assumptions -- it treats the allocation rate and the garbage fraction of the heap as constants. We would have liked to know how accurate the cost model is in cases where these assumptions don't hold. Evaluating various hybrid collectors on a set of benchmarks with varying allocation rates and garbage fractions would have answered these questions.

Regarding (2), we wondered how useful an abstract cost model is given that programmers would likely benchmark their application with different collectors if they were really concerned about garbage collection as a performance bottleneck.

Both of these concerns make us doubtful about the practicality of the proposed cost model. While it is an elegant abstraction and we appreciated the way the authors used it to compare their collectors, it would have been nice to see quantitative support for their comparisons.

# Connections to state of the art
An interesting thread of the discussion led some folks to the idea that the programmer
could play a bigger role in garbage collection, instead of the more abstract interface
that the mainstream paradigm provides. If a language implementation breaks
down some of the existing abstractions, collection could be more "domain-specific",
letting the compiler know which algorithms and hybrids to use on particular data structures
or sections of memory, how often to collect, or generally providing useful compile-time
guarantees.

A great example was the idea that standard library data structures could come
with programmer-facing guarantees on the GC's behavior. In this language, the user
might explicitly choose to use specific data structures because they have the 
guarantee of being reference-counted, for example, avoiding larger pauses. 

One other instance of breaking existing abstractions, this time relying slightly more on 
the programmer, is the idea that there should be a fundamental separation between 
"regions of memory" and "objects to free". In particular, the user should be able to 
operate on some allocated space with the guarantee that the collector will not 
collect objects in that space until directed by the programmer. For example,
if the program is operating on a graph, and all nodes are constantly active,
it would be wasteful to increment and decrement reference counts for every
change in the graph (or, indeed, mark-and-sweeping every so often); instead, 
there could be pointers into and out of the _space as a whole_, clearly outlining
when to free this large chunk of space. 

(Side note: a while after this point was brought up, it occurred to us that this 
is close to what a programmer would do to manually manage memory. It seems buggy
to free individual nodes during processing, so they might just free the whole
section when they're sure they've finished their computation. This goes back to
a meta-discussion-point about how there's still GC research to be done to fill the gap
between automatic, GC-managed memory and manual memory management.)
