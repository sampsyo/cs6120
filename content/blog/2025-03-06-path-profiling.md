+++
title = "Efficient Path Profiling Paper"
[extra]
latex = true
bio = """
  Allen, Kabir, Noah, and Simon were students at Cornell University in the Spring of 2025 -- at that time they diligently studied compilers under the tutelage of [Adrian Sampson](https://www.cs.cornell.edu/~asampson/)
"""
[[extra.authors]]
name = "Allen Wang"
[[extra.authors]]
name = "Kabir Samsi"
[[extra.authors]]
name = "Noah Schiff"
[[extra.authors]]
name = "Simon Bertron"
+++

This blog post summarizes the class discussion on the paper [Efficient Path Profiling](https://dl.acm.org/doi/10.5555/243846.243857) by Thomas Ball and James R. Larus.

### Background

**Overview of Profiling**

To understand the motivation of the paper, one can first take a closer look at what _profiling_
means in the context of control-flow graph analysis. When performing a program analysis over a control-flow graph,
key insights can be ascertained by looking at the most frequently visited paths through the control-flow graph.
The process of counting occurrences of different paths through the graph is referred to as path profiling.

In recent times, the strongest use case for profiling in programs has been _profile-guided optimization_
– a tool by which one can optimize programs based on what paths in a control-flow graph are the most frequently visited. 

**Edge Profiling**

One of the most popular profiling tools has been _edge profiling_ by which one can determine
the most frequently visited edges in a control-flow graph.
Edge profiling for a while served as the primary tool by which one could analyze well the "hottest" paths of a program.

However, there are a number of cases in which edge profiling just isn't sufficient in providing a strong enough
analysis for profile-guided optimization – furthermore, often making wrong predictions. This issue has been known for a while,
but has been tolerated as alternative, more accurate forms of profiling have historically come with a higher overhead.  

As such an example, one more accurate form of profiling which analyzes larger portions of the graph, is _path profiling_.
As its name would suggest, rather than looking purely at edges within the control flow graph and examining those which are the most frequently visited,
it instead looks at entire acyclic routines (paths) throughout the graph, and analyzes those which are traversed the most frequently.

An interesting analysis, benchmarking these two forms of profiling, can be found in [this paper](https://dl.acm.org/doi/pdf/10.1145/268946.268958) (also authored by Thomas Ball).

**Motivation for Efficient Path Profiling**

With this background in mind, this paper endeavors to, in its own words, demonstrate that "accurate profiling is neither complex nor expensive".
It endeavors to motivate path profiling over edge profiling, and addresses the concerns over overhead costs by presenting an efficient algorithm that negates this issue.

### Summary ###

The key contribution of this paper is an algorithm for path profiling. The core of the algorithm is a technique for assigning integers to edges in a control flow DAG such that

1. every path through the DAG gives a unique edge sum
2. the largest possible edge sum for any path through the DAG is 1 less than the total number of possible paths through the DAG (this is the smallest possible maximum edge sum)

_**Note:** the paper uses the term DAG and I have adopted it here, but the paper actually assumes stronger conditions than DAG-ness -- in particular, it assumes unique ENTRY (source) and EXIT (sink) nodes in the DAG_

The algorithm for assigning integers to edges is straightforward:

```
foreach v in reverse-topological-order(DAG):
    NumPaths(v) = IsExit(v) ? 1 : sum(NumPaths(w) for w in v's children)
    edge_weight = 0
    foreach w in v's children:
        weight(v,w) = edge_weight
        edge_weight += NumPaths(w)
```

The idea is that if you (a vertex in the DAG) have assigned some ordering to your outgoing edges and edge _e_ comes before edge _f_ in that ordering, any path that takes edge _e_ will have a lesser total sum than any path taking edge _f_.

Of course, the control flow graphs of many interesting programs are not DAGs. So the paper gives a technique for extending the algorithm to general control flow graphs. The key idea is to remove backedges (in the depth-first search sense of the term) and replace them with fake edges that can be more easily instrumented.

In particular, if $v \rightarrow w$ is a backedge, when you remove that edge you add two new edges: $ENTRY \rightarrow w$ and $v \rightarrow EXIT$. This process turns general control flow graphs into DAGs (and preserves the unique ENTRY and EXIT vertices). The resulting encoding does not distinguish _all_ paths through the graph (there are infinite distinct paths through a cyclic graph) but it does distinguish between some important paths (namely, paths that take 1 pass through a loop vs paths that take multiple passes).

The paper includes other details on placing instrumentation code for fake edges, placing instrumentation code efficiently, storing counts of path sums in arrays vs. hash maps, and selecting registers for storing the active path sum, but the key contribution is the algorithm described above.

### Analysis ###

The algorithm presented in this paper is interesting on pure theoretical grounds. It is always informative to find a minimum of anything, so the minimal encoding presented is valuable as a pure mathematical object. As a practical matter, the picture is slightly less clear.

It turned out (in the evaluation section) that many programs have too many paths to store path counts in an array, so the profiler presented in the paper often had to use a hash map instead.
This resulted in a somewhat bimodal distribution of profiling overheads: while the average profiling overhead across all programs was 31%, roughly bucketing the profiled programs into "low path count" and "high path count" (<10% hashed paths and >10% hashed paths in Table 1 in the paper, respectively) groups gave an average overhead of 16% for the low path count group and 43% for the high path count group.
There is even more variance in the overheads attributable to the number of instructions in the basic blocks of the CFG.
So while this profiling algorithm is certainly low-overhead _for a path profiling algorithm_, the observed overhead is highly dependent on the program being profiled.

This program dependence makes it tricky to argue about whether this algorithm is "low-overhead enough" for certain use cases.
An interesting use case to consider that was brought up during the class discussion of this paper is JIT compilers.
We asked ourselves "is this algorithm so low-overhead that we would feel comfortable running it on an actively running user program in a JIT compiler?"
For programs where the overhead of this algorithm is near 5% my answer is pretty comfortably "yes", but for programs where the overhead is close to 100% my answer is probably "no" or at least "very rarely".
Since it is not clear how well per-program profiling overhead can be statically predicted, this algorithm might be limited to uses cases where the worst-case overhead of 100% or higher can be tolerated.

Besides the path profiling algorithm, the paper's main argument is in favor of path profiling itself, arguing that it produces substantially better profiles than edge profiling.

The paper mostly successfully argues that path profiling produces better profiles.
The authors measure their own path profiles against path profiles produced from edge profiles in two ways.
One is by comparing the number of full executed paths that each method correctly predicts.
Path profiling obviously scores 100% on this metric, while edge profiling rarely breaks 50%.
The other method is by comparing the longest mistake-free path prefix that edge profiling produces to the length of the full executed paths.
The reported result is that edge profiling generally guesses the first 50-75% of a path correctly.
The biggest flaw in these two metrics is that they are not very holistic.
They ignore the fact that a distribution over edge frequencies gives an implied distribution over path frequencies that could be compared to the "true" distribution of path frequencies collected by the path profiler.
However, overall, this data paints a convincing picture that the information produced by path profiling is richer and more detailed.

The paper's deeper flaw is that it fails to take the final step and argue that the richer information leads to better optimized programs.
The authors don't do any experiments where the path and edge profiles are used to separately optimize the profiled programs.
Such an experiment would have completed their argument that path profiling is worth the extra cost.

Of course there was not really space to do a deeper analysis on edge vs. path profiling in this paper. Which is likely why that analysis was performed separately in the Ball paper from the background section. Although, disappointingly, even that paper doesn't include any experiments on actually optimizing programs with path and edge profiles.

### Role in History

This paper was the first to offer a performant algorithm for path profiling, so almost all of the other work on path profiling uses the paper's algorithm. The question here then just rounds to asking "what is the role of path profiling in history?"

The authors mention that path profiling could be potentially used for performance tuning, profile-guided optimization, and test coverage. For test coverage, it doesn't really see any use in practice for the reasons mentioned in [the discussion](https://github.com/sampsyo/cs6120/discussions/487) (too much performance cost for not enough benefit over edge profiling). For performance tuning and profile-guided optimization, it seems like path profiling sees some use in non-real time applications, but I'm not sure how often they're used in practice. Interestingly, profile-guided optimization tools like [these](https://research.facebook.com/file/900986544473313/Improved-Basic-Block-Reordering.pdf) [two](https://arxiv.org/abs/1810.00905) block layout optimizers that might be able to benefit from path profiling don't use it. Optimistically, this suggests that path profiling offers more utility than block or edge profiling, but that it can be difficult to make use of that utility. Pessimistically, this suggests that path profiling doesn't offer much more than block or edge profiling. The Codestitcher authors do mention that the information path profiling offers is "excessive" for code layout optimization, which I guess counts towards both viewpoints depending on how you want to think about it. For similar reasons as in profile-guided optimization, path profiling is not used in practice for real-time use cases like just-in-time compilation. Most of them also rely on simpler techniques like counting function calls or branch paths in order to determine when to patch in code. However, some other reasons why path profiling might be difficult to use here are the 30% performance drop and the difficulty of effective interprocedural optimization.

#### Other Papers

One interesting use-case for path profiling was in [this paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Holmes20-20Effecitve20Statistical20Debugging20via20Efficient20Path20Profiling.pdf), which uses path profiling to find which paths correlate with failures and suggests functions to instrument for debugging. [These](http://www.diag.uniroma1.it/delia/papers/oopsla13.pdf) [two](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Preferential20Path20Profiling20-20Compactly20Numbering20Interesting20Paths.pdf) papers also extend the original algorithm by adding compact numberings for a subset of interesting paths and handling loop iterations.
