 +++
title = "SIMD Divergence Optimizations"
extra.bio = """
  Philip Bedoukian is a 3rd year PhD student in ECE. His research focuses on reconfigurable hardware.
"""
extra.author = "Philip Bedoukian"
+++

The code used in this blog post is hosted [here](https://github.com/pbb59/bril/tree/proj2).

## Problem Statement

Parallel programming models must make tradeoffs between productivity and performance. Coarse-grain parallelization models like Single-Program Multiple-Data (SPMD) allow for high productivity, but aggressively generate vector instructions when cheaper scalar instructions would suffice. Conversely, fine-grain parallelization models such as C with SIMD intrinsics (C+SIMD) compiles to conservative scalar instructions by default, but requires additional programmer effort to manually insert vector instructions.

Divergence optimization seeks to provide the best-case performance of C+SIMD while maintaining the productivity of SPMD. The SPMD front-end still aggressively generates vector instructions, but a middle-end pass statically identifies unnecessary vector instructions and converts them into more efficient scalar instructions. One can do this conversion when each work-item/lane/thread in the vector instruction does the same computation. In the [literature](https://dl.acm.org/citation.cfm?id=3314902), divergence analysis has been shown to improve execution time by 1.5% on average for real GPU programs.

In this blog, we describe our implementation and empirical evaluation of divergence optimizations at the middle-end of the compiler stack.

## The SPMD Programming Model

SPMD is a highly productive parallel programming model with massive market share (ex. CUDA, OpenCL, OpenGL). In SPMD, a programmer describes parallelization at a coarse-grain level (i.e. at the level of an entire program). This contrasts a standard C+SIMD programming model which requires a fine-grained specification of when parallelization is desired (i.e. at the single-instruction level). One can think of the SPMD model as a higher level of abstraction than the SIMD model: in certain processors, SPMD will be compiled down to the SIMD model. 

While the coarse-grain specification of parallelization provides productivity, it also aggressively generates vector instructions. An obvious SPMD compilation procedure would generate a vector instruction for every instruction in the SPMD program. However, not every instruction needs to be parallelized. There are often values unknown at compile time, but constant across each parallel work-item. For example, memory indices and loop indices might be constant across lanes. In these cases, scalar instructions are optimal. A scalar instruction performs one operation for a group of parallel lanes and can later broadcast the single value to future vector instructions.

## Target Architecture

### Hardware

We target a simplified version of the GPU architecture described in [IGC](https://dl.acm.org/citation.cfm?id=3314902). Each core contains a single ALU with a vector length of four as well as scalar, vector, and predicate register files. A SIMD instruction will use operands from the vector register and run on all lanes of the ALU. Predicate registers can be used to mask off certain lanes of the vector instructions when there is control flow. Scalar instructions will use a scalar register and only a single lane of the ALU. A scalar instruction and an equivalent vector instruction will complete in the same amount of time, but a scalar instruction will consume less energy than the vector instruction. Less dynamic energy is required: to (1) read a scalar register than a vector one and (2) use only a single lane of the ALU rather than every lane. We will save energy anytime we can replace a vector instruction with an equivalent scalar one.

Although we target a specific architecture, every SIMD (Intel Integrated GPUs) and SIMT (nVidia and AMD Discrete GPUs) architecture can benefit from divergence optimizations.

### ISA

Predication was added to Bril to target the aforementioned hardware building upon the vector instructions added in our Project 1. A new instruction `vcmp` writes to a predicate register. A predicate register and its complement can be optionally specified before a vector instruction to mask off certain lanes of the ALU. Finally, two vector registers can be merged using the mask from a predicate register with a `vphi` instruction. An alternate implementation could remove the merge instruction and write to the same register directly at different indices. However, the former approach works better with the SSA format.

```python
# initialize vectors
  va: vector = ...;
  vb: vector = ...;

# generate the predicate
  p0: pred = vcmp va vb;

# ignore lanes based on predicate mask
  (p0)  vc0: vector = vadd va vb;
# ignore lanes based on complement of predicate mask
  (!p0) vc1: vector = vsub va vb;

# merge lanes back together
  vc: vector = vphi p0 vc0 vc1;

```

Each vector instruction supported by the interpreter is enumerated along with a description below.

|     Vector Instruction    | Description |
| ------------- | ------------- | 
| vadd  | c[i] = a[i] + b[i]  |
| vsub  | c[i] = a[i] - b[i]  | 
| vmul  | c[i] = a[i] * b[i]  | 
| vdiv  | c[i] = a[i] / b[i]  | 
| s2v  | c = (v0,v1,v2,v3)  | 
| s2vb  | c = (v0,v0,v0,v0)  | 
| v2s  | c = v[i]  | id |
| gather  | c[0,1,2,3] = mem[a0,a1,a2,a3]  | 
| scatter  | mem[a0,a1,a2,a3] = a[0,1,2,3]  | 
| vload  | c[i] = mem[i]  | 
| vstore  | mem[i] = a[i]  | 
| vcmp  | pred = a[i] == b[i]  | 
| vphi  | c[i] = pred ? a[i] : b[i]  |

## Divergence Analysis

Divergence analysis statically determines whether a vector instruction has redundant lanes of computation. In the following code, if `vec0` and `vec1` are vectors with the same value in each index, then the vector add will do the **exact** same work in each lane of the ALU. It would be much more efficient to do a single scalar (single-lane) `add` instruction instead.

```python
# initialize vectors
  vec0: vector = (v0, v0, v0, v0);
  vec1: vector = (v1, v1, v1, v1);

# add vectors
  vec2: vector = vadd vec0 vec1;

```

An instruction is assumed to be convergent (not divergent) by default. We traverse the dataflow graph forwards and mark an instruction as divergent if the following conditions are met. Our algorithm is based on the descriptions in [these](https://dl.acm.org/citation.cfm?id=3314902) [papers](https://ieeexplore.ieee.org/document/6113840).

|     Condition    | Description |
| ------------- | ------------------- |
| Instruction is `s2v` | A different scalar value is loaded into each index of a vector register |
| Instruction is `vload`  | Unknown values are loaded into each element of a vector register |
| Any data dependency is divergent  | An incoming edge in the dataflow graph is already divergent due to one of the previous conditions |

It's possible that during runtime some vectors marked as divergent might turn out to be convergent. For example, a `vload` may load in contiguous elements with the same values. However, there is no way to optimize for this case statically.

## Divergence Optimizations

### Instruction Swapping

Once we know which instructions are divergent and which are not, we can optimize the code on an instruction-by-instruction basis. In the previous Bril example, every vector instruction is convergent. Therefore we can swap each vector instruction with a more energy-efficient scalar instruction. The optimization is shown below. 

```python
# initialize vectors -> initialize scalars
  vec0_s: int = id v0;
  vec1_s: int = id v1;

# add vectors -> add scalars
  vec2_s: int = add vec0_s vec1_s;

```

We implement a 'swap table' that matches a vector instruction with a functionally equivalent scalar instruction. An alternate design would be to annotate each original scalar instruction with a vector length and just change the vector length instead of doing a swap. Our swap table is given below along with a description of each instruction reproduced from above.

|     Vector Instruction    | Description | Scalar Instruction |
| ------------- | ------------- | ------------- |
| vadd  | c[i] = a[i] + b[i]  | add |
| vsub  | c[i] = a[i] - b[i]  | sub |
| vmul  | c[i] = a[i] * b[i]  | mul |
| vdiv  | c[i] = a[i] / b[i]  | div |
| s2v  | c = (v0,v1,v2,v3)  | id |
| s2vb  | c = (v0,v0,v0,v0)  | id |
| v2s  | c = v[i]  | id |
| gather  | c[0,1,2,3] = mem[a0,a1,a2,a3]  | lw |
| scatter  | mem[a0,a1,a2,a3] = a[0,1,2,3]  | sw |
| vload  | c[i] = mem[i]  | Can't optimize |
| vstore  | mem[i] = a[i]  | Can't optimize |
| vcmp  | pred = a[i] == b[i]  | Can't optimize |
| vphi  | c[i] = pred ? a[i] : b[i]  | Can't optimize |

Notably, we can't optimize across `vload` and `vstore` instructions because different memory addresses are always accessed. However, in certain cases a `gather` and `scatter` can access the exact same memory location if the address vector is the same for each. In this case, the access will be redundant, which will waste memory energy and potentially execution time. Even though we perform the `scatter`/`gather` optimization in the compiler, it is likely that the hardware implementation would also detect this case and avoid the redundant accesses.

### Vector Regeneration

An instruction swap could create a register type mismatch between the result of the optimized instruction and future vector instructions that use that result. For this reason, a second pass is added to the optimization algorithm. After the scalar instructions have been created, we traverse each instruction in program order and detect when a vector argument points to a scalar register. Upon detection, an `s2vb` instruction is generated to effectively cast a scalar value to a vector value. The faulting instruction argument is then updated to the new vector value produced by this instruction.

The benefits of replacing a vector with a scalar outweighs the additional `s2vb` instruction. For example, a vector instruction with length four consumes three more ALU ops than a scalar instruction while an additional `s2vb` only consumes a single ALU op (we assume single-op broadcast). The overall benefit is then two ALU ops worth of energy savings.

### Predication Removal

Predicated vector instructions can also be simplified even in the case of a divergent predicate value. Every lane that is active in the vector instruction may still perform redundant work. Consider the Bril example below. The predicate `p0` is divergent because it's input `vec2` and `vec3` are divergent. However, the predicated vector instructions `vec4` and `vec5` are convergent because their inputs are convergent.

```python
# convergent vectors
  vec0: vector = (0,0,0,0);
  vec1: vector = (1,1,1,1);
# divergent vectors
  vec2: vector = (0,1,2,3);
  vec3: vector = (1,0,2,3);
# predicate p0 is divergent
  p0: pred = vcmp vec2 vec3;
# however both predicated computations are convergent
  (p0) vec4: vector = vadd vec0 vec0;
  (!p0) vec5: vector = vadd vec1 vec1;
  vec6: vector = vphi p0 vec4 vec5;

```

Thus, the code inside the predicate can be optimized, and the predicate can be removed because there are no longer lanes to mask out. The values still need to be merged afterwards according to the predicate to produce a result vector.

```python
# convergent vectors -> scalars
  vec0_s: int = const 0;
  vec1_s: int = const 1;
# divergent vectors
  vec2: vector = (0,1,2,3);
  vec3: vector = (1,0,2,3);
# predicate p0 is divergent
  p0: pred = vcmp vec2 vec3;
# convergent predicated code -> scalar instructions
  vec4_s: int = add vec0_s vec0_s;
  vec5_s: int = add vec1_s vec1_s;
# need to regenerate vectors to do the merge
  vec4_s_v: vector = s2vb vec4_s;
  vec5_s_v: vector = s2vb vec5_s;
  vec6: vector = vphi p0 vec4_s_v vec5_s_v;

```


## Evaluation

### Correctness

We test the correctness of the optimizations using [Turnt](https://github.com/cucapra/turnt). Turnt verifies both the code produced by the optimizations and the functionality (using the output of `print` instructions in the Bril code). We design six tests to check correctness. The tests are enumerated in the table below.

|     Test      | Description | Expected Optimization
| ------------- | ----------- | ---------- 
| vvadd | `vload` followed by a `vadd` | None, due to `vload` |
| Unique scalars  | Unique values written to vector (`s2v`), then `vadd` | None, due to unique values |
| Redundant scalars  | Redundant values written to vector, then `vadd` | All instructions should be scalar |
| Partially divergent  | Both convergent and divergent instructions | Optimize convergent instructions and add `s2v` as needed before divergent instructions |
| Predication - Unique   | Divergent predicated vector instructions | None, all divergent |
| Predication - Redundant | Convergent predicated vector instructions | Optimize convergent instructions with divergent predication and remove predication when possible.

### Performance

#### Metric

Our evaluation metric on the imaginary hardware is the number of ALU ops required by the program. Generally, each vector instruction consumes four ALU ops and each scalar instruction consumes a single ALU op. In this model, a scalar instruction is exactly four times as energy efficient as a redundant vector instruction. We argue that this is a good proxy metric for energy consumption if only the dynamic energy consumption of the ALU is considered and no other parts of the processor are considered (like memory access and on-chip network).

#### Benchmarks

We evaluate the effectiveness of the divergence optimizations on synthetic benchmarks. We take benchmark inspiration from the examples in [these](https://ieeexplore.ieee.org/document/6494995) [papers](https://hal.inria.fr/hal-00909072v2/document). All benchmarks have a 2D loop nest. We vectorize over each outer loop and unroll over each inner loop because we do not support most control flows. We manually unroll the inner loop twice for each benchmark as it allows us to get a sense of the dynamic ALU ops without actually running the program. The number of ALU ops for the baseline and optimized version of each benchmark is shown in the table below.

|     Benchmark    | Description | Baseline Ops | Optimized Ops | Improvement (%) |
| ------------- | ------------| ------------- | ------------ | ---------- |
| FIR  | 2D FIR filter | 60 | 53 | 12
| FIR-pred | 2D FIR filter with single conditional | 87 | 81 | 7
| Synthetic | Sum of (a[outer] * b[inner]) / c[inner] | 53 | 39 | 26

The optimization does lead to improvement in the number of ALU Ops for the listed benchmarks. It's hard to say exactly what a fair baseline would be because we don't know what a SPMD front-end would actually emit. For example, loads that aren't contiguous use `gather` in our baseline. In the case where each address is the same (i.e. indexed by the inner loop iterator), the `gather` can be turned into a `lw`. It's possible that this is obvious enough for a SPMD front-end to do automatically. We don't have a SPMD to Bril compiler, so we can't truly quantify a realistic improvement.


## Shortcomings

These are things that we didn't do, but would have improved the implementation and empirical results.

### SSA

The code must be in SSA form to perform divergence analysis on programs with arbitrary control flow. A control dependence can be converted into a data dependence with a `phi` instruction. These data dependencies fit naturally into the dataflow algorithm used in divergence analysis.

We were not able to implement transformations to and from SSA although we did successfully implement a `phi` instruction. To work around this limitation, we manually wrote our tests and benchmarks in an SSA-like form.

### Synthetic Benchmarks

Our benchmark selection was weak for two reasons. First, as described above, we had to manually code in SSA form which made programming in Bril meticulous. The second challenge was that we could not use a high-level language to create Bril programs. The current TypeScript front-end does not support vector instructions nor the SPMD model. 

## Conclusion

We implemented divergence analysis and optimizations based on that analysis. We focused on swapping expensive vector instructions for cheaper scalar instructions when the vector instruction did redundant work. We quantify our optimization by comparing the number of ALU Ops executed by the un-optimized baseline and optimized version. Our results show an overall reduction in ALU Ops.