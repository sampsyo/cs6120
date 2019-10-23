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

### Implementation details


### Experiment Results

