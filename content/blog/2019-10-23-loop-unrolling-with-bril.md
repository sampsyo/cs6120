+++
title = "Loop Unrolling with Bril"
extra.author = "Shaojie Xiang & Yi-Hsiang Lai & Yuan Zhou"
extra.bio = """
  [Shaojie Xiang](https://github.com/Hecmay) is a 2nd year ECE PhD student researching on programming language and distributed system. 
  [Yi-Hsiang (Sean) Lai](https://github.com/seanlatias) is a 4th year PhD student in Cornell's Computer System Lab. His area of interests includes electronic design automation (EDA), asynchronous system design and analysis, high-level synthesis (HLS), domain-specific languages (DSL), and machine learning. 
  [Yuan Zhou](https://github.com/zhouyuan1119) is a 5th year PhD student in ECE department Computer System Lab. He is interested in design automation for heterogeneous compute platforms, with focus on high-level synthesis techniques for FPGAs.  
"""
+++

## Project Report 2: Loop Unrolling with Bril

One key idea behind compiler optimization is to exploit the parallelism within a program. The most straightforward method is by optimizing loops. There exist many loop optimization techniques, including loop unrolling, reordering, tiling, pipelining, etc. Each technique can bring different effects to the given program. In this project, we focus on finding the pros and cons brought by loop unrolling with [Bril](https://github.com/sampsyo/bril) programs. We implement two methods for unrolling, which are static analysis and dynamic profiling, respectively.

### Methodology 

#### Input Specification

To make the analysis easier, we make the following assumptions for our input programs.

#### Loop Identification

The first step is to identify loops from a Bril program. In other words, we need to find **back-edges** within a control-flow graph. A back-edge is defined as a directed edge pointing from basic block A to basic block B where B dominates A. [(Additional readings on basic blocks and finding loops)](https://www.csl.cornell.edu/courses/ece5775/pdf/lecture06.pdf)

#### Static Analysis

#### Dynamic Profiling

The key idea of dynamic profiling is to estimate the loop status by simulating the loop execution. Since the loop trip count may depend on different conditions, it will hard to infer the loop bound without the information from actual execution environment. For example, induction variable being udpated indirectly can make the unrolling problem complicated. An intuitive way of simplifying the problem is to proflie the loops before unrolling. Here we extend the data flow analysis pass to support dynamic profiling, i.e. we maintain a status table to track the update history for each variables and . By refering to the table and locate the update history of the back-edge related instructions and basic blocks, we can easily infer various information needed, e.g. trip count, induction variable as well as loop stride. Here to simplify the analysis, we assume all loops to be unrolled are natural loops, namely the program can only enter the loop from the header. Under the assumption that the loop bound is invariant in multiple runs, the basic blocks in the loop body will be repliacted for trip bound times. Back edges (i.e. jmps) and condition check statements will be removed form originla bril program.

### Implementation details

#### Dynamic Profiling 

The dynamic profiling based unrolling pass consists of two parts mainly: loop status analysis, program structure re-construction. In the loop status analysis part, we created a variable table to track the defition and usage for each varible. The table is used as a look-up table for loop analysis and basic block replication in later stages. The variable table can be extended to record the data dependency between different varibles, and this information can be used to realize loop normalization, copy propagation and such. With the update information for each variables, we then iterate all the back edges in the control flow graph and unroll the loops accordingly. Note that the back edges are stored in a dictionary and need to be processed in order to make sure the loops can be unrolled correctly (e.g. for nested loops, the inner most loop needs to be unrolled first). In the last step, we replicate the loop body and remove the condition branch instructions as well as the back edges to construct the new unrolled program.

### Experiment Results

#### Dynamic Profiling Evaluation
The equivalence of the unrolled code can be verified with bril intepreter. The pass can run successfuly in different test cases including consecutive loops and double loops. In this section we mainly focus on the performance perspective. To evaluate the performance between original and unrolled programs, we converted the bril code from json representation to c using the [Bril C back end](https://github.com/Checkmate50/bril) from Dietrich Geisler. And then compile the C program, run it for 10000 times in with bash loop to measure the total time consumption. 

We construct a vector add program with large trip count numbers (2000), which can benefit loop unrolling and out-of-order execution of modern processors. The orginal and unrolled program takes 9.111s and 8.341s (measured with linux time utility in usr time) for 10000 execution.
