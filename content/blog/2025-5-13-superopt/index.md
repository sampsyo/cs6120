+++
title = "Technology Mapping with Egraphs"
[extra]
bio = """
  Arnav Muthiyan is a <!--TODO -->.<br>
  Neel Patel is a first year PhD student interested in computer architecture and systems.<br>
"""
[[extra.authors]]
name = "Arnav Muthiyan"
[[extra.authors]]
name = "Neel Patel"
+++

## Introduction
### E-graphs and Equality Saturation
E-graphs efficiently represent equivalence classes of expressions. This makes them useful for superoptimization, where, given an input program, we seek to find the sequence of optimizations that emits the *best* program. The example below shows an arithmetic expression a * 2 / 2 represented as an e-graph.

<!--- TODO -->

In *equality saturation*, we apply pattern-based *rewrites* to repeatedly produce equivalent expressions. Each rewrite grows the e-graph, without losing the previous versions of the expression. Upon saturation, an e-graph will have undergone enough rewrites to reach a fixed point where it encodes all possible expressions simultaneously.
<!--- -->
Equality saturation has found applications in compiler optimization and, recently RTL synthesis. The E-graphs Good (egg) library provides a fast and extensible implementation of equality saturation, enabling the use of E-graphs in more applications.
<!--- -->
The benefit of equality saturation is that it avoids the problem of finding the optimal order in which to apply optimizations (the phase-ordering problem) by encoding all possible optimizations within the e-graph. The catch is that a separate procedure, called *extraction*, is required to actually select the best term from the e-graph according to a user-provided cost function.
<!--- -->
For simple cost functions, it is sufficient to apply a greedy, bottom-up, extraction procedure. However, for more complex cost functions, extracting an optimal solution has been proven to be NP-hard. Currently, Egg implements a [heuristic greedy extractor](https://github.com/egraphs-good/egg/blob/v0.10.0/src/extract.rs) and an [exact extractor](https://github.com/egraphs-good/egg/blob/v0.10.0/src/lp_extract.rs) which formulates extraction as a mixed integer linear programming problem and solves it using the [COIN-OR](https://github.com/coin-or/Cbc) Branch-and-cut solver. The former has been shown to select terms with suboptimal costs, while the latter does not scale well to larger problems, leading to a poor scalability-quality tradeoff [[Cai et al. 2025](https://www.csl.cornell.edu/~zhiruz/pdfs/smoothe-asplos2025.pdf)].

### Superoptimization for Technology Mapping
<!--TODO: Brief intro on what RTL and ASIC technology mapping is.-->

<!--TODO: Explain what the state-of-the-art in technology mapping i.e., ABC and heuristic algorithms -->


<!--TODO: Then explain our contributions
1) Integration of good_lp library into egg, enabling the use of a wider range of extraction techniques
2) Comparison of different ILP solvers and greedy extraction for superoptimization of RTL designs using a standard cell library -- TODO (what standard cell library?)
-->
###

## Superoptimizing RTL

### Specifying an RTL Design
<!-- TODO: Example of a circuit written in LUTLang -->

### Equality Saturation
<!-- TODO: Show how the circuit is represented as an egraph-->

<!-- TODO: Show the rewrite rules and how the rewrites are applied to the egraph  -->

### Design Extraction

<!-- TODO: Show and expalin the extraction techniques (currently implemented in Egg) that we can use to extract the design from the egraph
1) Greedy Extraction
2) ILP Extraction using branch-and-cut
  a) [COIN-OR](https://coin-or.github.io/Cbc/intro)
  b) [SCIP]()
-->

## Results