+++
title = "Memory Optimization and Profiling for MLIR-Based HeteroCL"
[extra]
bio = """
  [Hongzheng Chen](https://chhzh123.github.io/) is a first-year CS PhD student at the Computer Systems Laboratory, Cornell University. His research interests include domain-specific languages, compiler optimization, and heterogeneous computing systems.
  [Niansong Zhang](https://www.zzzdavid.tech/) is a first-year ECE PhD student at Cornell.
  [Jiajie Li](https://tonyjie.github.io/) is a first-year ECE PhD student at Cornell.
"""
latex = true
[[extra.authors]]
name = "Hongzheng Chen"
link = "https://chhzh123.github.io/"
[[extra.authors]]
name = "Niansong Zhang"
link = "https://www.zzzdavid.tech/"
[[extra.authors]]
name = "Jiajie Li"
link = "https://tonyjie.github.io/"
+++

<!-- Proposal: https://github.com/sampsyo/cs6120/issues/311 -->
<!-- For the main project deadline, you will write up the project’s outcomes in the form of a post on the course blog. Your writeup should answer these questions in excruciating, exhaustive detail:
* What was the goal?
* What did you do? (Include both the design and the implementation.)
* What were the hardest parts to get right?
* Were you successful? (Report rigorously on your empirical evaluation.) -->

## Introduction
[HeteroCL](https://github.com/cornell-zhang/heterocl)[^1] is a programming infrastructure composed of a Python-based domain-specific language (DSL) and a compilation flow that targets heterogeneous hardware platforms. It aims to provide a clean abstraction that decouples algorithm specification from hardware customizations, and a portable compilation flow that compiles programs to CPU, GPU, FPGA, and beyond.

## Reuse Buffer
The original HeteroCL paper only talks about how to generate reuse buffers for simple 2D convolutional kernel, while real-life applications generally have high-dimensional arrays with complex access patterns. In this project, we try to extend the idea of reuse buffer in the MLIR framework to support more applications. In this section, we will first discuss the high-level design of reuse buffer and details the implementation.

### Design
The basic idea of reuse buffer is to reuse data between adjacent loop iterations so that the number of off-chip memory accesses can be reduced. If without special notification, we suppose the traversal order is the same as the memory access order in this report.

The applications that have data reuse are mostly *stencil* kernels, so we follow SODA[^3] to give a similar definition. Consider a point $\mathbf{x}=(x_0,x_1,\ldots,x_m)\in\mathbb{N}^m$, a $n$-point stencil window defines a set of offsets $\{\mathbf{a}^{(i)}\in\mathbb{N}^m\}_{i=1}^n$ that describe the distance from $\mathbf{x}$. By adding the offset to a specific point, we can get the actual points $\{\mathbf{y}=\mathbf{x}+\mathbf{a}^{(i)}\}_{i=1}^n$ of a stencil in a specific iteration. The value of these points will be reduced (e.g., mean, summation) to one value in the output tensor.

We further define the *span* $s_d$ of a dimension $d$ as the largest distance between two stencil points along that dimension.

$s_d=\max_{i}(\mathbf{a}^{(i)}_d)-\min_i(\mathbf{a}^{(i)}_d)+1$

As the following example in Fig. (a), we consider a five-point stencil

$\{\mathbf{a}^{(i)}\}^{5}_{i=1}=\{(0,1),(1,0),(1,1),(1,2),(2,1)\}$

colored with red, and the span of each dimension is $s_0=s_1=2$.

![](reuse_buffer.png)

Declaring this kernel in HeteroCL is easy. Users can leverage the declarative API `hcl.compute()` to explicitly write out the computation rule as follows.
```python
def test_stencil():
    hcl.init(dtype)
    A = hcl.placeholder((width, height))

    def stencil(A):
        B = hcl.compute(
            (width - 2, height - 2),
            lambda i, j: A[i, j + 1]
            + A[i + 1, j]
            + A[i + 1, j + 1]
            + A[i + 1, j + 2]
            + A[i + 2, j + 1],
            name="B",
            dtype=dtype,
        )
        return B

    s = hcl.create_schedule([A], stencil)
    B = stencil.B
```

From the memory access pattern above, we can see that when the stencil window is moved from the previous iteration (blue grids in Fig. (a)) to the next iteration in the same row (red grids in Fig.(a)), there exist two replicated data in these two iterations (highlighted with dash line). If all the data are loaded from off-chip memory, it may cause contention and introduce large memory access overheads. To exploit this data reuse opportunity, HeteroCL provides a push-button primitive `.reuse_at()` for users to declare reuse buffers. By specifying the dimension, users can exactly control the location of reuse buffers.

The best case to generate minimal size reuse buffer is to generate small retangular strips whose total length is equal to the reuse distance of the stencil [^2]. In the five-point stencil example, this requires generate buffer covering elements from $(0,1)$ to $(2,1)$. The total size of the buffer is $4+6+3=13$. However, this may introduce complicated control logic when implementing on FPGA. Thus, for simplicity, we use *rectangular hull* of the stencil as the reuse buffer, which is labeled as red frame in Fig. (a). We call it *window buffer*. The shape of this buffer is $[s_0,s_1,\ldots,s_m]$. We can always reuse data when the buffer moves along the same row.

While this works well for one dimension, when the stencil window moves from the end of previous row to the front of the next row, there will be no elements that can be reused, which may incur extra latency to load data from off-chip memory. A traditional way to tackle this problem is leveraging another load process to prefetch the data, so the reuse buffer can always be prepared with data. This method again complicates the control logic and requires extra prefetching function to work concurrently. To be consistent with the API that HeteroCL proposed, we can further create hierarchical reuse buffers to hide memory access overheads. As shown in Fig. (b), we can create a *line buffer* to reuse data along the column dimension. Basically, only one element needs to be fetched from off-chip memory to line buffer in each iteration. The original elements in the line buffer need to be shifted up for one grid as depicted in blue arrows. After the elements are loaded to line buffer, they are further copied to window buffer as shown in the red arrows. In this way, the data loading pipeline can work perfectly without stall. Finally the kernel can simply use the indices in $\{\mathbf{a}^{(i)}\}^{5}_{i=1}$ to access the elements in the window buffer.

The programming interface is also easy-to-use, users only need to attach the following two lines of code to their schedule. Our compiler will automatically generate the buffer and implement the reuse logic.

```python
LB = s.reuse_at(A, s[B], B.axis[0])
WB = s.reuse_at(LB, s[B], B.axis[1])
```

## Write Buffer

## Roofline Model

## Experiments
<!-- A major part of your project is an empirical evaluation. To design your evaluation strategy, you will need to consider at least these things:
* Where will you get the input code you’ll use in your evaluation?
* How will you check the correctness of your implementation? If you’ve implemented an optimization, for example, “correctness” means that the transformed programs behave the same way as the original programs.
* How will you measure the benefit (in performance, energy, complexity, etc.) of your implementation?
* How will you present the data you collect from your empirical evaluation?
Other questions may be relevant depending on the project you choose. Consider the [SIGPLAN empirical evaluation guidelines](https://www.sigplan.org/Resources/EmpiricalEvaluation/) when you design your methodology. -->

## Conclusion and Future Work

## References
[^1]: Yi-Hsiang Lai, Yuze Chi, Yuwei Hu, Jie Wang, Cody Hao Yu, Yuan Zhou, Jason Cong, Zhiru Zhang, "*HeteroCL: A Multi-Paradigm Programming Infrastructure for Software-Defined Reconfigurable Computing*", FPGA, 2019.
[^2]: Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, Oleksandr Zinenko, "*MLIR: Scaling Compiler Infrastructure for Domain Specific Computation*", CGO, 2021.
[^3]: Yuze Chi, Jason Cong, Peng Wei, Peipei Zhou, "*SODA: Stencil with Optimized Dataflow Architecture*", ICCAD, 2018.
[^4]: Louis-Noel Pouchet, Peng Zhang, P. Sadayappan, Jason Cong, "*Polyhedral-Based Data Reuse Optimization for Configurable Computing*", FPGA, 2013