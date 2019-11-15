+++
title = "Profile Guided Basic Block Reordering"
extra.author = "Qian Huang and Horace He"
extra.bio = """
Qian Huang is a junior undergraduate studying CS and Mathematics.

Horace He is a senior undergraduate studying CS and Mathematics
"""
+++

Improving instruction layout is a key way to improve instruction cache utilization and avoid branch missing panelty. Ideally, we would like instructions to be executed mostly sequentially, such that instructions are already fetched in cache and branch predictor can also have high accuracy. One simple way to achieve this is basic block reordering. The de facto code positioning technique is provided in [Profile Guided Code Positioning](http://pages.cs.wisc.edu/~fischer/cs701.f06/code.positioning.pdf). In this project, we implemented and tested one variant of the algorithm descirbed in this paper with LLVM.


# Design Overview

## LLVM 

LLVM is an ahead-of-time compiler that allow us to experiment with compiler optimization as well as analyze programs in general. It has three major components: 
- The front end that takes in the source code and transforms it to LLVM intermediate representation(IR). In this project, we just used Clang untouched as the front end.

- The passes that transform IR to IR through various optimizations. To experiment with Basic Block Reordering, we simply added it as one LLVM pass as guided by this [blog](https://www.cs.cornell.edu/~asampson/blog/llvm.html).

- The backend that generates final machine code. In this project, we just used Clang untouched as the back end.

## Profile Guided Optimization

Profile guided optimization uses profiling imformation to improve program runtime performance. This means that we run the program with its typical inputs and record the frequency of each branch being taken. Consider one simple example:
 
```c
int a = atoi(args[1]);
int b = 0;
if (a > 5) {
    b = 4;
} else {
    b = 6;
}
printf("%d", b);

```

which will be compiled to LLVM IR as

```
  ...
  %13 = icmp sgt i32 %12, 5
  br i1 %13, label %14, label %15, !prof !31

14:                                               ; preds = %2
  store i32 4, i32* %7, align 4
  br label %16

15:                                               ; preds = %2
  store i32 6, i32* %7, align 4
  br label %16

16:                                               ; preds = %15, %14
  ...
  ret i32 %19
}

```
This simply put the blocks in the original order. If we know that the input `a` is almost always smaller than 5, then we could move 15 and 16 block before 14, such that the instruction cache would not need to fetch 14. In the case where 14 branch has much more instruciton, fetching 14 instead of 15 can lead to instruction cache miss. 

In this project, we used the [build-in profile generation via instrumentation tool in Clang](https://clang.llvm.org/docs/UsersManual.html#profiling-with-instrumentation) to collect profiles. 

## Block Reordering Algorithm

We implemeted one variant of algo2 in Profile Guided Code Positioning as one LLVM pass. The frequencies of each branch in conditional jump and switch instrucitons are used as edges weight. We also experiemnted with marking weight of unconditional branch to be infinity.

The algorithm first identifies frequently executed "chains", i.e. paths of basic blocks. It starts by setting all basic blocks as individual chains. Then the edges are iterated from highest weight to lowest. For each edge, if it is connecting tail and head of two chains, the chains will be merged. We implemented this by maintaining the start of chains and the next block relationships in hashmaps. Then we iterate through all the block terminators to collect edges with weights and rank them by weights. Finally we merge the chains if and only if the edge is jumping from the end of a chain to the start of a chain.

After the chains are identified, they are arranged sequentially based on weights over edges cross chains. In our implementation, we decide the next chain to put in the way that the current chain has one branch with largest weight point to the start of it. 


## Evaluation

### Initial Testing
Initially we tested our pass on simple programs with only one conditional branch, where one branch is more frequently executed than the other. Our pass successfully rearranged the more used pass to directly after the block before branching. We also tested on more complicated branching scenarios. 

### Benchmark Testing

We also evaluated the pass on [PARSEC](https://parsec.cs.princeton.edu/). Due to machine and time constraints, we only managed to run streamcluster programs with profiles provided by running the simulated inputs. We compared clang with only our optimization pass against clang with no optimization. Unfortunately, the running time and branch missing rates are not significantly improved:


We think this is because the profiling information we collected is not enough to reflect the actual workload, since we are only using small simulated input for profiling. Clang profiling also does not provide unconditional branch frequency. Also the branch missing rate is already pretty low even without any optimization. 

### Extensions and Improvement

Ideally we would like to implement more complicated position ordering and have more rigorous testing set up, including collecting more profiles and find better benchmark suit. 
