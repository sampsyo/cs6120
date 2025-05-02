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
<!--TODO: Brief intro on what RTL and ASIC technology mapping is.-->

<!--TODO: Explain what the state-of-the-art in technology mapping i.e., ABC and heuristic algorithms -->

<!--TODO: Then explain superoptimization and egraphs (provide an illustration). Explain why they are useful for RTL optimization as well as other optimization problems with a large search space (e.g., compiler optimizations) -->

<!--TODO: Then explain our contributions
1) Integration of good_lp library into egg, enabling the use of a wider range of extraction techniques
2) Comparison of
-->
###

## Superoptimizing RTL

### Specifying an RTL Design
<!-- TODO: Example of a circuit written in LUTLang -->

### Equality Saturation
<!-- TODO: Show how the circuit is represented as an egraph-->

<!-- TODO: Show the rewrite rules and how the rewrites are applied to the egraph  -->

### Design Extraction

<!-- TODO: Show and expalin the currently possible extraction techniques used to extract the design from the egraph
1) Greedy Extraction
2) ILP Extraction using branch-and-cut
  a) [COIN-OR](https://coin-or.github.io/Cbc/intro)
  b) [SCIP](https://www.scipopt.org/)
-->

## Results


<!---

### Technology Mapping
Technology Mapping with Boolean Matching, Supergates and Choices: https://people.eecs.berkeley.edu/~alanmi/publications/2005/tech05_map.pdf
* The task of technology mapping in standard-cell logic synthesis is to express a given Boolean function as a network of gates chosen from a given standard-cell library so that some objective function, such as total area or delay, is optimized.
* cut-based techniques found in technology mapping for FPGA look-up tables can be adapted to work for standard cell libraries using Boolean matching


### ILP Algos:
* [cbc](https://en.wikipedia.org/wiki/Branch_and_cut#:~:text=Branch%20and%20cut%20is%20a%20method%20of,the%20algorithm%20is%20called%20cut%20and%20branch)
  *

### Example:
https://www.cs.cornell.edu/courses/cs6120/2023fa/blog/hcl-amc/
https://github.com/neel-patel-1/cs6120/tree/2023fa/content/blog/2023-12-09-hcl-amc
-->