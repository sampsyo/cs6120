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
Solutions to this problem thus focus on counting other execution properties dynamically, such as the number of times each control flow graph (CFG) edge is taken.

Such an algorithm, however, does not allow unique construction of paths, and cannot accurately predict the path profile of a given execution.
Consider, for instance, the edge profiles shown in Figure *INSERT HERE*.
These measurements _both_ compose to produce the path profile shown in Figure *INSERT HERE*.
Neither measurements, however, predict the path _ABCDEF_ as the most taken, despite 

Indeed, according to this paper, it was long thought that accurate path profiling