+++
title = "Software Simulation for Data Streaming in HeteroCL"
extra.author = "Shaojie Xiang & Yi-Hsiang Lai"
extra.bio = """
  [Shaojie Xiang](https://github.com/Hecmay) is a 2nd year ECE PhD student researching on programming language and distributed system. 
  [Yi-Hsiang (Sean) Lai](https://github.com/seanlatias) is a 4th year PhD student in Cornell's Computer System Lab. His area of interests includes electronic design automation (EDA), asynchronous system design and analysis, high-level synthesis (HLS), domain-specific languages (DSL), and machine learning. 
"""
+++

With the pursuit of higer performance under physical constraints, there has been an increasing deployment of special-purpose hardware accelerators such as FPGAs. The traditional appraoch to program such devices is by using hardware description languages (HDLs). However, with the raising complexity of the applications, we need a higher level of abstraction for productive programming. C-based high-level synthesis (HLS) is thus proposed and adopted by many industries such as Xilinx and Intel. Nonetheless, in order to achieve high performance, users usually need to modify the algorithms of applications to incorporate difference types of hardware optimization, which makes the programs less productive and maintainable. To solve the challenge, recent work such as [HeteroCL](http://heterocl.csl.cornell.edu/) proposes the idea of decoupling the algorithm from the hardware customization techniques, which allows users to efficiently explore the design space and the trade-offs. In this project, we focus on extending HeteroCL with data streaming support by providing a software simulation runtime. Experimental results show that ...

### Why Data Streaming?

Unlike traditional devices such as CPUs and GPUs, FPGAs do not have a pre-defined memory hierarchy. Namely, in order to achieve better performance, the users are required to design their own memory hierarchy, including data access methods such as streaming. In this project, we focus on the streaming between on-chip modules. The reasone that we are interested in the cross-module streaming is that it introduces more parallelism to the designs. To be more specific, we can use streaming to implement task-level parallelism. We use the following example written in HeteroCL to illustrate the idea of streaming.

```python


```

In this example, ``kernel1`` takes in one input tensor ``A`` and writes to two output tensors ``B`` and ``C``. Then, ``kernel2`` and ``kernel3`` read from ``B`` and ``C`` and write to ``D`` and ``E``, respectively. We can see that ``kernel2`` and ``kernel3`` have no data dependence and can thus be run in parallel. Moreover, these two kernels can start as soon as they receive an output produced by ``kernel1``. To realize such task-level parallelism, we can replace the intermediate results ``B`` and ``C`` with data streams. We illustrate the difference between before and after applying data streaming with the following figure.


### Data Streaming in HeteroCL

