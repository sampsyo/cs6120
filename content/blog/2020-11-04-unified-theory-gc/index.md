+++
title="Unified Theory of Garbage Collection"
[[extra.authors]]
name = "Mark Moeller"
[extra]
bio = """
  Mark Moeller is a first-year graduate student in Computer Science at Cornell
  interested in compilers and formal foundations of programming languages.
"""
latex = true
+++

## Background

This week's paper, [A Unified Theory of Garbage Collection][utgc] \(Bacon,
Cheng, and Rajan (IBM)\), appeared in OOPLSA 2004.

There are two classical algorithms for automatic garbage collection, tracing
(also called mark and sweep), and reference counting, which were both first
developed in the 1960s.  [Java's generational garbage collector][java] is a
typical example of a tracing collector, and [CPython's garbage
collector][python] uses reference counting---but as this paper suggests, the
reality is that these distinctions are certainly muddier in practice.

The two paradigms are associated important performance tradeoffs. In particular,
the mark and sweep approach adds significant pauses to program progress, since
in the simplest approach one must pause execution to 'trace' all live objects
before reclaiming unused space. Reference counting on the other hand does
not have long pauses during execution, but rather pays for its overhead 
incrementally. Reference counting's major drawback is the difficulty presented
by cycles appearing the link structure, which prevent objects' reference counts
from ever reaching 0.

## Highlights

Bacon et al. begin by presenting the two fundamental garbage collection
algorithms, reference counting and tracing, as dual views of the same underlying
structure. This should hit you a bit like hearing that butter and I Can't
Believe It's Not Butter have the same deep structure.

The deep comparison they make between the two paradigms is developed in a number
of ways:

* **Duality of operations:** Tracing traverses the live
  objects on the heap in a technically analogous way to how reference
  counting traverses dead objects. 

* **Algorithm structure:** They make this more concrete by giving pseudocode algorithms
  for each that really are strikingly similar.

* **Fixed point forumation:** They give a formal presentation of garbage
  collection from an abstract standpoint. In this abstract view, the goal of garbage
  collection is to compute a fixed-point of a system of references where nodes'
  reference counts accurately reflect in-edges from either (i) members of the
  root set, or (ii) objects whose reference count is nonzero. The fixed-point
  here could be imagined as the result of some iterate-to-convergence approach.

They go on to demonstrate that a whole continuum of hybrid approaches between
the two not only make sense in principle but in fact really are already in use
in practice. Here is a list of the the garbage collection strategies in *rough*
order from most mark-n-sweepy to most-reference-county:

* **T: Tracing.** This is the classic tracing algorithm. The algorithm must
  traverse the link graph of the heap to 'mark' objects live. The authors point
  out that tail recursion can reduce the space cost of the traversal when singly
  linked structures are present.

* **GT: Tracing Generational Collection.** This form of collection uses a
  "nursery" to improve the time spent on collection. The idea is that recently
  allocated objects are more likely to be freed earlier, so we can spend our
  garbage-collection time more efficiently by looking at newer objects more
  often.

* **Older First.** (Sketched only). In this approach, the oldest generation
  is traced first instead of the nursery.

* **GCH: Generational with Counted Heap.** Run normal tracing of the nursery,
  but do reference counting on the mature remaining space of the heap.

* **Train Algorithm.** (Presented but not analyzed). While this approach was
  definitely interesting, and they drive home that it fits into their hybrid
  model (car ordering provides implicit reference counts), I must admit I was not
  able to glean much intuition about why the approach would be worth
  implementing.

* **CC: Reference Counting, Trial Deletion.** This is mostly reference counting,
  but with some tracing of certain "candidates" mixed in to detect cycles (i.e.
  check if a node's reference count is artificially high due to circular
  references; evidently the case we care about is when a reference count is
  non-zero purely due to circular references, in which case its memory should be
  reclaimed).

* **CD: Deferred Reference Counting.** Do reference counting, but save overhead
  by not counting mutations of root references. Add items
  to a Zero Count Table (ZCT) when they are decremented to zero. References from the
  "root set" (a concept borrowed from tracing meaning "references into the heap
  from outside of the heap") can keep ZCT entries alive.

* **CT: Reference Counting, Tracing Backup.** Use reference counting without
  attempting to detect cycles. Occasionally do a full tracing collection to
  clean up cyclic garbage.

* **C: Reference Counting.** This is the classic reference counting algorithm. 
  They give a really slick way to reduce space
  overhead for traversing the links by using the space guaranteed to exist in
  the dead objects themselves.

The paper concludes with a mathematical analysis of the costs of the
above collection strategies. The authors point out that the analyses are not 
big-O, so these costs are actually "real", up to the accuracy of the constants
for a particular system (and I guess also the constancy of those constants?).

The analyses focus on computing the formulas for: 
* $\phi$, the frequency of garbage collection
* $\kappa$, the cost of an individual collection, and
* $\tau$, the total cost of collection, which is generally $\phi \cdot \kappa$
  plus the cost of mutation overhead

The computation of $\phi$ and $\kappa$ are kept separate, presumably because
they quanitify the essence of the classic tradeoffs between tracing (large
pauses) and reference counting (continuous overhead). Thus, by investigating the
$\phi$ and $\kappa$ values of an approach one might be considering on the system
one has in mind, one could evaluate where along the spectrum of hybrids the
approach lies (on that system).


## Merits of the paper

The duality between tracing and reference counting itself is a both mind-bending
and extremely well developed.  Given that the paper is really just offering this
insight about a nice way to categorize existing garbage collection methods, its
organization is quite nice.  That is, they give the deep comparison abstractly
in several different ways, then completely flesh out how existing hybrids
combine the two flavors.

The section of the paper that has the most potentional to be practically
important is the cost analysis in section 7.  Having never constructed a garbage
collection system for a whole language, I do not know whether a
compiler designer would actually use these formulas to tune a garbage collector
(even with knowledge of their particular domain) since they would have to
validate any decisions against benchmark performance anyway.


## Impact on state of the art

This paper does not technically offer any new approaches to storage reclamation
(although to be sure, its authors have certainly done that in other works).
On the other hand, it does seem like one of those papers that changes how one
thinks about the topic forever. Of course, the extent to which it has done that
for people in the field is certainly hard to know. It does seem, at least, that
plenty of people spend a lot of time tuning the hybridization of their
garbage collectors, so for them this paper gives a really nice
framework for thinking about how to do that. (For those curious about more
recent developments in this hybrid GC realm, there is [Immix][immix] and its
reference-counting-infused descendant, [RCImmix][rcimmix]).

They indicate their motivation for the cost analysis was actually to aid in
development of dynamic construction of garbage collection algoirthms. That
sounds like an interesting avenue for future work!

## Discussion Questions
1. The seminal papers they cite for reference counting and tracing were both
   from 1960. This one was 2004. Does it seem like this observation took a
   while? If so, why did it?
2. This paper doesn't give any new algorithms or approaches for garbage
   collection. In what ways does the mere observation of this duality have value
   to the community?
3. Can you give any situations or domains where using either (relatively) pure
   tracing or pure reference counting is advantageous over one of these
   "optimized hybrid approaches"?

[utgc]:    https://dl.acm.org/citation.cfm?id=1028982
[python]:  https://devguide.python.org/garbage_collector/
[java]:    https://www.oracle.com/webfolder/technetwork/tutorials/obe/java/gc01/index.html
[rcimmix]: https://dl.acm.org/doi/10.1145/2509136.2509527
[immix]:   https://dl-acm-org.proxy.library.cornell.edu/doi/10.1145/1375581.1375586
