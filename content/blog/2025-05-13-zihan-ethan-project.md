+++
title = "Final Project: Parallel Dataflow Analysis"
[extra]
bio = """
  Ethan Uppal Cornell CS '27
  Zihan Li Cornell CS '25
"""
[[extra.authors]]
name = "Ethan Uppal"

[[extra.authors]]
name = "Zihan Li"
+++

## Evaluation

> [!NOTE]
> Some of these questions are redundant in the context of both sections and thus their answers will be too.

**Where will you get the input code you’ll use in your evaluation?**

We will use both the core benchmarks (a restriction imposed by our decision to keep a simple custom Bril representation) and generate our own CFGs through fuzzing.

**How will you check the correctness of your implementation? If you’ve implemented an optimization, for example, “correctness” means that the transformed programs behave the same way as the original programs.**

We will use a sequential implementation based on one we already tested for a previous CS 6120 assignment (and therefore have high confidence that it is correct).
If the parallel result agrees with the sequential result on all inputs, we will consider our implementation "correct".

**How will you measure the benefit (in performance, energy, complexity, etc.) of your implementation?**

TODO

**How will you present the data you collect from your empirical evaluation?**

TODO

## Experience Report

**What was the goal?**

Build a parallel dataflow solver for analyses supporting bitset optimization.

**What did you do? (Include both the design and the implementation.)**

- We built a simple representation of Bril programs that assigns variables numbers instead of strings to make it easy to apply the bitset optimization.
- We implemented a simple generic dataflow solver for analyses supporting bitset optimization.
- Using the aforementioned framework, we implemented live variable and reaching definition analysis.
- We applied Tarjan's algorithm to decompose a CFG into a DAG of SCCs. DAGs, of course, naturally lend themselves to parallelism.
Thus, we would apply the sequential implementation to each SCC and schedule dependent jobs for each SCC in a thread pool following the DAG dependencies.
- We had plans for more intelligent load balancing that were scrapped due to time constraints.

**What were the hardest parts to get right?**

Correctness in the implementation of the parallel dataflow solver, in particular, managing successors and predecessors between and within components.
We tried 

**Were you successful? (Report rigorously on your empirical evaluation.)**

We tested on the core benchmarks and wrote a fuzz tester to automatically generate "interesting" CFGs to test with.
By test, we mean we implemented a sequential oracle solver and used it to gauge the correctness of the parallel solver.
