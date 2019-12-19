+++
title = "ML based phases ordering with LLVM"
extra.author = "Qian Huang and Horace He"
extra.bio = """
Qian Huang is a junior undergraduate studying CS and Mathematics.

Horace He is a senior undergraduate studying CS and Mathematics.
"""
+++

In this project, we continued to experiment with ML based phases ordering as in project 2, using c++ programs and LLVM instead. Compared project 2 with Bril, there are alot more infastructure supports for this project: more avaiable optimization passes, actual hardware performance counters and actual benchmarks. These allowed us to explore this topic more.

# Design Overview

## Analysis

We first analyzed how much the ordering of optimization passes affects program performances. We selected the 270 optimizations used in llvm `-O2` level as the set of optimizations we analyzed cross the whole project. We performed all optimizations on the streamcluster program in PARSEC benchmark.

figure

At each data point, we uniformly randomly select n optimization passes with replacement (so there can be duplicate paths) and run 10 times to obtain a range of performance. From the figure, we see that the ordering of optimizations do affect the overall program performance significantly. In fact some order of optimizations even make it significantly worse. As the number of passes used increases, we can see that the optimal performance improvement indeed increases but saturated around ....

## Methods

### Hill Climbing

One popular approach is Hill Climbing: selecting the optimization gives most performance boost at each step. However, it can be really expensive to rebuild and run 270 times at every step, while also rerun the program multiple times to reduce performance variance. In the experiment, we sampled 5 optimizations at each step instead of trying all, and only used one run performance.

### Model

We built a simple linear regression model to predict how much one optimization pass will improve over the performance of current program, based on a set of performance counters we selected from ones provided by [likwid-perfctr](https://github.com/RRZE-HPC/likwid). During inference time, we will run the model to approximate hill climbing algorithm: select the next optimization pass that gives best performance improvement, based on the score given by the model, until we reach maximum number of passes limit or there are no more optimizations that will improve the performance.

### Training models

To collect data points of program performance counters, optimization pass and performace differentce, we random sampled optimizations sequences as in the analysis but with random length. We then collect the performace and performance counter values without the last optimization as features. The performance counters we used are:

We then fit a linear regression model to predict normalized the performance improvement. 

## Evaluation

### Random testing at each step

### Benchmark Testing

We also evaluated the pass on [PARSEC](https://parsec.cs.princeton.edu/). Due to machine and time constraints, we only managed to run streamcluster programs with profiles provided by running the simulated inputs. We use `parsecmgmt` to run the benchmarks, and report the branch misses as well as the total runtimes. We compared clang (no optimizations) with only our optimization pass against clang (no optimizations). Unfortunately, there isn't a significant improvement in either the branch misses or the runtime. We suspect there is at least some improvement in streamcluster, considering the drastically lowered standard deviation and modestly reduced branch misses.

| Benchmark                              |No optimizations     | Our pass            |
|----------------------------------------|---------------------|---------------------|
| Streamcluster (branch-misses 10ks)     | 3326.58094 +/- 35.9 | 3312.82899 +/- 1.45 |
| Streamcluster (runtime)                | 15.389 +/- 1.414    | 15.316 +/- 1.704    |


We think this is because the profiling information we collected is not enough to reflect the actual workload, since we are only using small simulated input for profiling. Clang profiling also does not provide unconditional branch frequency. In addition, the branch miss rate is already pretty low even without any optimizations. Even without any compiler optimizations, there are plenty of other optimizations in lower layers that likely allow for low branch-misses.

### Extensions and Improvement

Ideally we would like to implement more complicated position ordering and have more rigorous testing setup, including collecting more profiles and benchmarking on a much more exhaustive list of benchmarks.
