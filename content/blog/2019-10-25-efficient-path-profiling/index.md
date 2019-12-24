+++
title = "Efficient Path Profiling "
extra.author = "Dietrich Geisler"
extra.bio = """
  [Dietrich Geisler](https://www.cs.cornell.edu/~dgeisler/) is a 3rd year PhD student researching Language Design and Compilers.  Enjoys gaming and climbing.
"""
+++

## Introduction

Identifying path bottlenecks on program execution is critical when evaluating and optimizing program performance.  
Due to the exponential number of paths in large programs, however, counting the most frequented path in a given execution directly is intractable for any realistic program.
Solutions to this problem long focused on counting other execution properties dynamically, such as the number of times each control flow graph (CFG) edge is taken.

Such an edge-focused algorithm, however, does not allow unique construction of paths, and cannot accurately predict the path profile of a given execution.
The usual definition for the most taken path is defined greedily, where it is constructed as the series of maximally taken paths for each node.
Consider, for instance, the edge profiles, which _both_ compose to produce the path profile shown:

<img src="bad_paths.png" alt="drawing" width="400"/>

Neither measurements, however, predict the path _ABCDEF_ as the most taken, despite it being defined as such by the usual heuristic.
Indeed, it was long thought that this loss of accuracy was necessary when defining a sufficiently fast algorithm.

We will explore how Thomas Ball's and James R. Larus' paper [_Efficient Path Profiling_](https://dl.acm.org/citation.cfm?id=243857) define an efficient algorithm for dynamically computing information about path use.
This path profiling algorithm computes information about paths directly, and so does not suffer from the same loss of accuracy as approximations derived from edge use measurements.
As promised by the title, this algorithm is both memory efficient and fast enough to use when profiling realistic programs.
Experimentally, path profiling is shown to only have a 31% profiling overhead compared with edge profiling having a 16% overhead on a standard benchmark suite.

## The Algorithm for DAGs

We start by assuming profiling over a directed acyclic graph (DAG), but will later explore extending to loops.
Any DAG will have at least one node with no incoming edges, called entry nodes, and at least one node with no outgoing edges, called exit nodes.
We extend any CFG with a single entry and single exit node, thus making each unique.
The steps of the path profiling algorithm are as follows:
1. Assign a minimal number of integer values to edges such that the sum of integers along any path is unique.
2. Using a spanning tree, create and select instrumentation for computing the increment for each edge.
3. Collect the dynamic runtime profile.
4. Derive the executed paths based on the results and selected instrumentation.

### Assigning Edge Integer Values

The first step to path profiling is to assign edge integers such that the sum of edge integers along any path in the DAG results in a unique integer value.
There are many such assignments, but the minimal positive set can be given by reasoning about the number of paths each edge can take to reach the exit node.
Intuitively, calculating a unique integer based on paths works since each split creates a new path from the unique entry node and results in a distinct integer.
Formally, the value for each edge from _v_ to _w_ is constructed to be the sum of the number of paths from all previously examined nodes extending from _v_.
This result, for example, is the integer values for our sample CFG:

<img src="integers.png" alt="drawing" width="300"/>

### Creating Instrumentation

The unique paths given by these integers is not yet efficient counting instrumentation.
Counting instrumentation is based on incrementing a specific register whenever a given edge is passed; such writes can be expensive, so we select edges to minimize the number of writes while still producing correct results.
To construct counting efficiently, we must first identify the most taken edges (based on the edge weights computed earlier) and avoid incrementing when using those edges.
Specifically, we compute the maximum spanning tree with respect to edge weights.
All updates then only reason about those edges in the set of _chords_, the edges not in the maximum spanning tree.

Instrumentation for the graph (the code for updating register associated with the path) is then added for each edge in the chord based on the integers computed earlier.
Since each path of integers must be unique, the register selected by a given chord edge is simply given as the minimum integer increment since the last chord edge for any path.
This algorithm gives the instrumentation for the integers calculated on our sample CFG as follows, where edges with squares are chords:

<img src="instrumentation.png" alt="drawing" width="300"/>

### Regenerating Paths

Finally, after incrementing each register according to our instrumentation, we must recover the number of times each path was taken.
The integers calculated earlier give an intuition for reconstructing these values, since each integer path must have been unique.
We can then just calculate the integer value associated with each path by walking the path in the integer version of the CFG and use this to reference which register corresponds to each path.

### Extending to Cycles

We have assumed so far that every graph is acyclic; how can we extend this algorithm to work with cycles?
It turns out that evaluating each type of cycle and how it relates to an arbitrarily inserted enter or exit node provides sufficient information to construct our integer edge values as with a DAG.

Cycles added to a DAG can be thought of as _backedges_, edges which visit a node previously seen in the DAG.
Through replacing these backedges with information-carrying forward edges, the algorithm reconstructs a DAG from any cyclic graph.
The details of these edges rely on technical casework, so those interested in the details should read the EPP paper directly; however, it is sufficient for our overview to state that each backedge is replaced by forward edges from the entry node to its target, and from its source to the exit node.
This process can be summarized visually with this excellent diagram:

<img src="cycle.png" alt="drawing" width="300"/>

## Evaluation

To evaluate how realistic this approach can be, the authors built the PP tool and compared it with an existing edge profiling tool (QPT2).
Several optimizations were applied to PP to promote a realistic comparison, such as mapping local registers through the Executable Editing Library (EEL) and implementing hash tables to handle a large number of paths.
All experiments were run on the SPEC95 benchmark, which consists of a standard set of C and Fortran programs.

Experiments were run on a Sun Ultraserver, with 167Mhz UltraSPARC processors and a whopping 2GB of memory.
PP's overhead on this test suite averaged 30.9%, while QPT2's overhead averaged 16.1%.
Note that cache interference caused by profiling was not recorded; however, in general, programs with little hashing (few paths) had comparable PP and QPT2 overhead while programs with substantial hashing had a larger PP overhead.
The PP tool had perfect accuracy definition; in comparison, the QPT2 tool only averaged 37.9% accuracy, demonstrating the main strength of the PP approach.

These experiments show the relative power of this new algorithmic approach.
While the path profiling algorithm does cause some additional overhead, this was shown to be minimal compared to edge profiling algorithms; the gains in accuracy also speak for themselves.
This test suite is rather small, however, only consisting of 18 test programs, which makes generalizing these results somewhat difficult.
While the author's analysis of the results shows some thought, it almost seems they accepted that the relatively low overhead shown here indicates that the accuracy is simply worth the cost.
Note also that while these experiments only compared PP to one other tool; there are so few other profiling algorithms that this seems appropriate.
The QPT2 tool is noted to be the lowest overhead edge profiling tool; with the accuracy guarantees provided by PP, testing against other tools feels almost meaningless.

The complete timing results are provided below.
The authors also included a summary of accuracy results, which have been omitted for simplicity.

<img src="results.png" alt="drawing" width="300"/>

## Conclusion

The efficient path profiling paper introduced an algorithm showing that accurate path profiling _can_ be done quickly and efficiently.
This insight resulted in an algorithm that is, as best I can tell, still in use today.
Through proving the minimal properties of this path profiling approach, this paper seems to have resolved the direction of profiling and standardized the algorithm presented here.
