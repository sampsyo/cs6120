+++
title = "Technology Mapping with Egraphs"
[extra]
bio = """
  Arnav Muthiayen is a third year undergraduate student interested in computer architecture.<br>
  Neel Patel is a first year PhD student interested in computer architecture and systems.<br>
"""
[[extra.authors]]
name = "Arnav Muthiayen"
[[extra.authors]]
name = "Neel Patel"
+++

## Introduction
### E-graphs and Equality Saturation
Equality-graphs, or more simply, e-graphs, efficiently represent equivalence classes of expressions. This makes them useful for superoptimization, where, given an input program, we seek to find the sequence of optimizations that emits the *best* program, minimizing some user-provided cost function. E-graphs are directed graphs with edges from expressions (*e-nodes* in an e-graph), to a group of e-nodes (an *e-class* in an e-graph, where all the e-nodes contained are equivalent expressions). To efficiently maintain and merge these equivalence classes, e-graphs rely on a union-find data structure. The example below shows an arithmetic expression `a * 2 / 2` represented as an e-graph.

<img src="canonical-example-before.svg" alt="" width="33%">

In *equality saturation*, we apply pattern-based *rewrites* to repeatedly produce equivalent expressions. Each rewrite grows the e-graph, without losing the previous versions of the expression. So if two expressions always evaluate to the same result, they belong in the same e-class. Upon saturation, an e-graph will have undergone enough rewrites to reach a fixed point where it encodes all possible expressions simultaneously.

Using the previous example expression, `a * 2 / 2`, three (among many other) rewrite rules can be applied:
1. Division associativity
2. Constant folding
3. Multiplicative identity

As rewrite rules are applied to the expression `a * 2 / 2`, each transformation produces an equivalent form: first rewriting it as `a * (2 / 2)` using division associativity, then simplifying to `a * 1` via constant folding, and finally reducing to `a` using the multiplicative identity. Though the expression changes, all versions are equivalent. It's important to note that applying more rewrite rules, such as multiplicative commutativity or replacing `x * 2` with `x << 1`, doesn’t erase existing expressions. Instead, it expands the e-class by adding more equivalent expressions. The e-graph grows to represent several equivalent expressions, giving the optimizer more options to choose from.

<img src="canonical-example-after.svg" alt="" width="33%">

Equality saturation has found applications in compiler optimization and, recently RTL synthesis. The [E-graphs Good](https://egraphs-good.github.io/) (egg) library provides a fast and extensible implementation of equality saturation, enabling the use of E-graphs in more applications.
The benefit of equality saturation is that it avoids the problem of finding the optimal order in which to apply optimizations (the phase-ordering problem) by encoding all possible optimizations within the e-graph. The catch is that a separate procedure, called *extraction*, is required to actually select the best term from the e-graph according to a user-provided cost function.

For some simple cost functions, the optimal expression can be found by applying a greedy, bottom-up, extraction procedure, but, in general, the problem of extracting an optimal solution has been shown NP-hard. Currently, Egg implements two extractors.
The first is a [heuristic greedy extractor](https://github.com/egraphs-good/egg/blob/v0.10.0/src/extract.rs) which expresses the cost of an e-node as the aggregate cost of its children.
The second is an [exact extractor](https://github.com/egraphs-good/egg/blob/v0.10.0/src/lp_extract.rs) which formulates extraction as a mixed integer linear programming problem and solves it using the [COIN-OR](https://github.com/coin-or/Cbc) Branch-and-cut solver.

### E-graphs for Technology Mapping
The task of technology mapping in logic synthesis is to express a given Boolean function as a network of gates from a standard cell library (for ASICs) or programmable LUTs (for FPGAs) so that an objective function, such as total area, is optimized.

Traditional technology mapping tools like [ABC](https://people.eecs.berkeley.edu/~alanmi/abc/) rely on cut enumeration and dynamic programming to select optimal gate implementations. ABC supports both FPGA and ASIC targets and can operate on logic networks with structural choices—precomputed alternative implementations of subcircuits. Its recent addition of a priority-cut-based mapper improves performance by only considering the most promising cuts per node, reducing memory use and runtime. However, the mapper's effectiveness still depends heavily on the initial circuit structure, which may limit optimization potential.

E-graphs offer a more expressive alternative. Rather than selecting cuts locally, they grow a graph of equivalent expressions using rewrite rules. This enables the representation of many circuit topologies simultaneously, effectively enumerating structural choices upfront. After saturation (or a time limit is reached), extraction selects an implementation based on a cost model. Prior work has demonstrated the promise of this approach. [E-Syn](https://arxiv.org/pdf/2403.14242) integrates e-graph rewriting into a delay- and area-aware mapping, showing measurable improvements over standard AIG-based pipelines in delay and area savings. [ROVER](https://ieeexplore.ieee.org/iel8/43/10762795/10549954.pdf) applies e-graphs to RTL datapath optimization and uses ILP-based extraction to achieve up to 63% area savings. These results validate e-graphs as a competitive backend for logic synthesis, capable of exploring larger design spaces than traditional mappers.

### Contributions
In this project, we integrate multiple linear programming solvers into an e-graph-based electronic design automation (EDA) tool. By using the [good_lp](https://github.com/rust-or/good_lp) library to implement an exact extractor in the [egg](https://github.com/egraphs-good/egg) library, the EDA tool and users of egg's exact extractor can run extraction with these solvers.
We also evaluate the performance of the extractor in the EDA tool, which transforms designs specified in the verilog hardware description language into designs targetting FPGAs or ASICs.
Our good\_lp-based exact extractor is [available](https://github.com/neel-patel-1/egg) and can be used by any projects using the egg library.

## Optimizing RTL using E-Graphs

### Specifying an RTL Design

For this project, we use an EDA tool, written by Matt Hoffman, called `lvv`. lvv performs optimizations on a domain-specific, circuit-specification language called *LutLang*.
Here is a rough outline of the grammar defined by LutLang:
```
LutLang> ::= <Program> | <Node> | BUS <Node> ... <Node>

<Node> ::= <Const> | x | <Input> | NOR <Node> <Node> | MUX <Node> <Node> <Node>
            | LUT <Program> <Node> ... <Node> | REG <Node> | ARG <u64> | CYCLE <Node>

<Const> ::= false | true // Base type is a bool

<Input> ::= <String> // Any string is parsed as an input variable

<Program> ::= <u64> // Can store a program for up to 6 bits
```

lvv takes verilog as input, but converts it into LutLang before performing optimizations and then converts it back into verilog.

Below we give an example of a half-adder written in LutLang:

`(BUS (AND a b) (XOR a b))`


### Equality Saturation
An e-graph representation of the original design using boolean logic, is shown below:

<img src="simple_2_output_before.svg" alt="" width="20%">

The two-input gates are first transformed into programmable LUTs specified by three parameters: a truth table (numeric e-node), and two operands (a and b). During equality saturation, rewrite rules are applied until all possible designs are encoded in the e-graph.

<img src="simple_2_output_lvv_after.svg" alt="" width="33%">


### Design Extraction

During extraction, an optimal design is produced using either a greedy or exact (linear programming) method as explained in the [E-graphs and Equality Saturation](#e-graphs-and-equality-saturation) section.

To apply linear programming to the e-graph extraction problem, a set of constraints and objective function must be specified. A number linear-programming libraries implement algorithms to solve linear programming problems. The good\_lp rust crate was developed to make it easier to apply any of the [HiGHs](https://highs.dev/), [SCIP](https://scipopt.org/#scipoptsuite), [microlp](https://github.com/Specy/microlp/), and [COIN-OR Branch-and-Cut](https://github.com/coin-or/Cbc) algorithms to linear programming problems. With good\_lp, the application developer specifies the set of problem variables, a set of constraints, and an objective function to maximize/minimize.  We specify the problem using the formulation of [Yang et al.](https://arxiv.org/pdf/2101.01332). The problem of e-graph extraction can be formalized as follows:

Let:
- `i = 0, ..., N - 1` be the set of e-nodes in the e-graph.
- `m = 0, ..., M - 1` be the set of e-classes in the e-graph.
- `e_m` denote the set of e-nodes within e-class `m`: `{i | i ∈ e_m}`.
We introduce a binary integer variable `x_i` for each e-node `i`. A node `i` is selected if `x_i = 1`, and not selected otherwise.
#### Objective Function:
Each e-node is associated with a cost `c_i`. The objective is to minimize the total cost of the selected nodes:
#### Subject to:
1. `x_i ∈ {0, 1}` for all `i`.
2. `Σ x_i = 1` for all `i ∈ e_0` (root e-class).
3. For all `i ∈ h_i` and `m ∈ h_i`, `x_i ≤ Σ x_j` for all `j ∈ e_m`.
4. Acyclicity constraints: For all `i, m ∈ h_i`, `t_g(i) - t_m - c + A(1 - x_i) ≥ 0`.
5. Bounds on Acyclicity variables: `0 ≤ t_m ≤ 1`.

good\_lp will transform the problem specification into the implementation-specific data structures and method invocations to solve the problem. By writing our exact extractor using good\_lp, the implementation is agnostic to the solver backend and the problem variables, constraints, and objective function are easy to determine upon inspection.

## Results

We compare the number of LUTs in the designs extracted using greedy and ILP extraction on the [ISCAS85](https://sportlab.usc.edu/~msabrishami/benchmarks.html) design benchmarks. Since finding an exact solution quickly becomes prohibitive in terms of extraction time, (the size of the e-graph and corresponding linear programming problem gets too large), we incrementally increase the number of "rewrite iterations" until reaching 10 . This limits the size of the e-graph by restricting the number of times rewrites are applied. Some benchmarks/solvers were not able to complete even a single iteration within the 30 minute timeout threshold.

| Benchmark | No Optimization LUT Count  | Greedy LUT Count | Microlp LUT Count (# Rewrite Iterations) | Highs LUT Count (# Rewrite Iterations) | CBC LUT Count (# Rewrite Iterations) |
|-----------|----------------------------|------------------|------------------------------------------|----------------------------------------|---------------------------------------|
| c1355     | 96                         | 94               | DNF                                      | 96 (10)                                | 96 (5)                                |
| c17       | 2                          | 2                | 2 (10)                                   | 2 (10)                                 | 2 (10)                                |
| c1908     | 86                         | 85               | DNF                                      | 84 (8)                                 | 84 (5)                                |
| c2670     | 120                        | 119              | DNF                                      | 118 (10)                               | DNF                                   |
| c3540     | 265                        | 260              | DNF                                      | 264 (1)                                | DNF                                   |
| c432      | 51                         | 50               | DNF                                      | 50 (10)                                | 50 (4)                                |
| c499      | 90                         | 90               | DNF                                      | 90 (10)                                | 90 (5)                                |
| c5315     | 267                        | 266              | DNF                                      | 259 (6)                                | DNF                                   |
| c6288     | 520                        | 512              | DNF                                      | 515 (6)                                | 520 (2)                               |
| c7552     | 335                        | 325              | DNF                                      | 315 (7)                                | DNF                                   |
| c880      | DNF                        | DNF              | DNF                                      | DNF                                    | DNF                                   |

The times to solution are reported in the table below

| Benchmark | Greedy Time       | Microlp Time      | HiGHS Time        | CBC Time           |
|-----------|-------------------|-------------------|-------------------|--------------------|
| c1355     | 0.046134086       | DNF               | 640.92384104      | 230.738058229      |
| c17       | 0.011094708       | 4.205207809       | 0.00869014        | 0.007125629        |
| c1908     | 0.03812723        | DNF               | 1551.254409928    | 488.214537225      |
| c2670     | 0.042055477       | DNF               | 600.226674074     | DNF                |
| c3540     | 0.032852412       | DNF               | 112.26789725      | DNF                |
| c432      | 0.022252978       | DNF               | 860.208582938     | 603.27150813       |
| c499      | 0.023853089       | DNF               | 729.54756512      | 600.49411377       |
| c5315     | 0.047760014       | DNF               | 743.806362887     | DNF                |
| c6288     | 0.136245309       | DNF               | 608.710588015     | DNF                |
| c7552     | 0.045072024       | DNF               | 600.26766583      | DNF                |
| c880      | DNF               | DNF               | DNF               | DNF                |

*Takeaway:* We observe that there is a scalability, quality tradeoff between greedy and exact extraction. The latter requires orders of magnitude longer time to produce a solution, and can even produce worse solutions when limiting the number of rewrite iterations. As designs scale, it may not be feasible to run exact extraction to find the optimal design.

We also compare the area (µm²) of the designs extracted using greedy and exact extraction (this time only using the top-performing - HiGHs - solver) on the ISCAS85 verilog design benchmarks using a standard cell library. Synthesizing an ASIC design using exact extraction becomes prohibitive faster than synthesis targetting an FPGA due to the larger design search space. There is no "No Optimization" column in this chart. To convert to a standard cell library, a set of rewrite rules must be applied required to convert digital logic to standard cells. We also compare against the Synopsys commercial design compiler to show the design quality a tuned EDA tool can achieve.

| Bench   | Greedy Area   | Exact Area (Node Limit) (msynth) | Synopsys               |
|---------|---------------|----------------------------------|------------------------|
| c1355   | 317.87        | 415.23 (16000)                   | 254.56                 |
| c17     | 7.18          | 6.92 (2000)                      | 6.92                   |
| c1908   | 330.11        | 374.53 (8000)                    | 229.29                 |
| c2670   | 587.86        | 760.76 (8000)                    | 424.00                 |
| c3540   | 867.16        | 981.28 (8000)                    | 537.59                 |
| c432    | 183.27        | 176.36 (2000)                    | 107.46                 |
| c499    | 259.08        | 272.38 (4000)                    | 255.63                 |
| c5315   | 1373.61       | DNF                              | 813.43                 |
| c6288   | 2672.49       | DNF                              | 1239.83                |
| c7552   | 1760.10       | DNF                              | 913.18                 |
| c880    | 259.88        | 265.73 (4000)                    | 224.24                 |

The times to perform greedy and exact extraction are reported below. We use a 10-minute timeout for the solver, omitting results for which it is unable to produce a valid solution within the alloted time.

| Benchmark | Greedy Time       | Exact Time (msynth) |
|-----------|-------------------|---------------------|
| c1355     | 0.046134086       | 600.090794699       |
| c17       | 0.011094708       | 14.406110368        |
| c1908     | 0.03812723        | 600.034827982       |
| c2670     | 0.042055477       | 114.496902413       |
| c3540     | 0.032852412       | 72.048779133        |
| c432      | 0.022252978       | 14.406110368        |
| c499      | 0.023853089       | 0.853112572         |
| c5315     | 0.047760014       | DNF                 |
| c6288     | 0.136245309       | DNF                 |
| c7552     | 0.045072024       | DNF                 |
| c880      | 0.027521613       | 10.793451794        |

*Takeaway:* Greedy extraction cannot achieve the design quality of optimized EDA tools, but it is impractical to achieve high quality designs using exact extraction, due to its poor scalability.

## Challenges

The end goal of the project changed after the proposal. Initially, we aimed to implement an efficient extraction algorithm, called [SmoothE](https://www.csl.cornell.edu/~zhiruz/pdfs/smoothe-asplos2025.pdf), into the EDA tool used throughout this project. As a stepping stone towards an implementation, we decided to first implement a linear programming-based extraction algorithm.
Getting our LP solver implementation to emit correct results took more effort than expected. At the same time, the EDA tool had only recently begun to implement support for ASIC logic synthesis using a standard cell library and exact extraction had not yet been fully implemented and tested. For this reason we decided to focus on developing a correct implementation of LP extraction. Despite the less ambitious end goal, there were numerous challenges.

Long synthesis times for exact extraction made debugging challenging. Working with simple, fast-to-synthesize test cases is not enough. Simple test cases' e-graphs are not representative of complex designs with thousands of e-nodes, hundreds of thousands of constraints, and many cycles.
The size and complexity of logic synthesis for ASICs revealed the limitations of solver libraries. Errors from the underlying libraries were frequent. To address these, manual tuning of the problem was required -- we limited the size of the e-graph by restricting the number of total e-nodes and rewrite iterations.