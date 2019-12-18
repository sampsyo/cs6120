+++
title = "Software Simulation for Data Streaming in HeteroCL"
extra.author = "Shaojie Xiang & Yi-Hsiang Lai"
extra.bio = """
  [Shaojie Xiang](https://github.com/Hecmay) is a 2nd year ECE PhD student researching on programming language and distributed system. 
  [Yi-Hsiang (Sean) Lai](https://github.com/seanlatias) is a 4th year PhD student in Cornell's Computer System Lab. His area of interests includes electronic design automation (EDA), asynchronous system design and analysis, high-level synthesis (HLS), domain-specific languages (DSL), and machine learning. 
"""
+++

With the pursuit of higer performance under physical constraints, there has been an increasing deployment of special-purpose hardware accelerators such as FPGAs. The traditional appraoch to program such devices is by using hardware description languages (HDLs). However, with the raising complexity of the applications, we need a higher level of abstraction for productive programming. C-based high-level synthesis (HLS) is thus proposed and adopted by many industries such as Xilinx and Intel. Nonetheless, in order to achieve high performance, users usually need to modify the algorithms of applications to incorporate difference types of hardware optimization, which makes the programs less productive and maintainable. To solve the challenge, recent work such as [HeteroCL](http://heterocl.csl.cornell.edu/) proposes the idea of decoupling the algorithm from the hardware customization techniques, which allows users to efficiently explore the design space and the trade-offs. In this project, we focus on extending HeteroCL with data streaming support by providing cycle-inaccurate software simulation. Experimental results show that ...

### Why Data Streaming?

Unlike traditional devices such as CPUs and GPUs, FPGAs do not have a pre-defined memory hierarchy. Namely, in order to achieve better performance, the users are required to design their own memory hierarchy, including data access methods such as streaming. In this project, we focus on the streaming between on-chip modules. The reasone that we are interested in the cross-module streaming is that it introduces more parallelism to the designs. To be more specific, we can use streaming to implement task-level parallelism. We use the following example written in HeteroCL to illustrate the idea of streaming.

```python


```

In this example, ``kernel1`` takes in one input tensor ``A`` and writes to two output tensors ``B`` and ``C``. Then, ``kernel2`` and ``kernel3`` read from ``B`` and ``C`` and write to ``D`` and ``E``, respectively. We can see that ``kernel2`` and ``kernel3`` have no data dependence and can thus be run in parallel. Moreover, these two kernels can start as soon as they receive an output produced by ``kernel1``. To realize such task-level parallelism, we can replace the intermediate results ``B`` and ``C`` with data streams. We illustrate the difference between before and after applying data streaming with the following figure.


### Data Streaming in HeteroCL

The key feature of HeteroCL is to decouple the algorithm specification from the hardware optimization techniques, which is also applicable to streaming optimization. To specify streaming between modules, we use the primitive ``to(tensor, dst, src, depth=1)``. It takes in four arguments. The first one is the tensor that will be replaced with stream. The second one is the destination module and the third one is the source module. Finally, the users can also specify the depth of the stream. Currently, data stream is implemented with FIFOs. HeteroCL will provide other types of streaming in the future. Following we show how to specify data streaming with our previous example.

```python
```

### Software Simulation for Data Streaming

It is not enough with the programming language support only. We also need the ability to simulate the programs after applying data streaming. One way to do that is by using the existing HeteroCL back ends. Namely, we can generate HLS code with data streaming and use the HLS tools to run software simulation. Note that the software simulation here refers to cycle-inaccurate simulation. The reason why we only focus on cycle-inaccurate simulation is that to complete cycle-accurate simulation, we need to run through high-level synthesis, which could be time-consuming in some cases. We can see that the existing back ends require users to have HLS tools installed, which is not ideal for an open-srouce programming framework. Moreover, the users will need a separate compilation to run the simulation. Thus, in this project, we introduce a CPU simulation flow to HeteroCL by extending the LLVM JIT runtime. With this feature, users can easily verified the correctness of a program after introducing data streaming.

### Implementation Details

The key idea is to simulate data streaming with threads. In other words, each module will be executed using a single thread. We also implement a scheduling algorithm to decide the firing of a thread and the synchroniztion between threads. For streaming, we implement the streams by using one-dimensional buffers. We assign the size of a buffer according to the specified FIFO depth. Currently, we only provide blocking reads and blocking writes. Non-blocking operations will be left as our future work. In the following sections, we describe the algorithms and the implementation details.

#### Module Scheduling

The purpose of this algorithm is to schedule each module by assigning it with a timestep, which indicates the execution order between modules. Namely, modules that can be executed in parallel are assigned with the same timestep. Similarly, if two modules are executed in sequential, they are assigned with different timesteps. Note that the numbers assigned to two consecutive executions do not need to be continuous. Since each module is executed with a single thread, a thread synchronization is enforced between two consecutive timesteps.

To begin with, we first assign each module with a group number. Modules within the same group are executed in sequentail while modules in different groups can be executed in parallel. To assign the group number, we first build a dataflow graph (DFG) according to the input program. An example is shown in the following figure, where the solid lines mean normal read/write operations while the dotted lines refer to the read/write of data streams.

[Insert Figure]

After the DFG is built, we remove all the dotted lines. Then, we assign a unique ID to each connected components. This ID will be the group number. An example is shown below.

[Insert Figure]

Now, we can start the scheduling process by assigning the timestep to each module. We first perform a very simple as-soon-as-possible (ASAP) algorithm. Namely, the first module within each group will be assigned with timestep 0. After that, we assign the timestep of each module according to the data dependence. An example is show below.

[Insert Figure]

However, this is not correct because as we mention above, modules connected with streams should be run in parallel. Namely, they will share the same timestep. To solve that, we add one dotted line back at a time and correct the the timesteps. We also need to correct its successing modules accordingly.

[Insert Figure]

After all dotted lines are added, we finish our scheduling algorithm. Note that there exist cases where we cannot solve. For example, if two modules ``A`` and ``B`` are connected with a solid line, and the producer ``A`` streams to a module ``M`` while ``B`` also streams from ``M``, then there exists no valid scheduling according to our constraints. One potential way to solve that is by merging ``A`` and ``B`` into a new module ``A_B``. In this case, the streaming from/to ``M`` becomes an internal stream, which can be scheduled easily by assigning ``A_B`` and ``C`` with the same timestep.

#### Parallel Execution with Threads

After we assign each module with a timestep, we can start to execute them via threads. Before we execute a module with a new thread, we check whether all modules assigned with smaller timesteps are completed. In other word, we first check whether all modules assigned with smaller timesteps are fired. If not, we schedule the current module to be executed in the future by pushing it into a sorted execution queue. Then, if all modules with smaller timesteps are fired, we check whether they are finised. If not, we perform thread synchronization (e.g., by using ``thread.join()`` in C++). Finally, we need to execute the modules in the execution queue. Since the queue is sorted, we do not need to worry about new modules being inserted into the queue.

#### Stream Buffers

In this work, we implement the streams with buffers that act like FIFOs. Instead of actually popping from or pushing to the buffers, we maintain a **head** and a **tail** pointer for each buffer. The pointers are stored as integer numbers. The head pointer points to the next element that will be read from and the tail pointer points to the next element that will be written to. We update the pointers each time an element is written to or read from the buffer. We need to perform modulo operations if the pointer value is greater than the buffer size (i.e., FIFO depth). Since we may have two threads updating the pointers at the same time, we declare them as atomic numbers. Finally, we maintain a map so that we can access a stream according to its ID.

#### LLVM JIT Extension

To enable users with a one-pass compilation, we extend the existing LLVM JIT runtime in HeteroCL. It is complicated and hard to maintain if we implement both threads and stream buffers using only LLVM. Thus, we implement them with C++ and design an interface composed of a set of functions. For instance, we have ``BlockingRead``, ``BlockingWrite``, ``ThreadLaunch``, and ``ThreadSync``. Then, inside our JIT compiler, we call the functions by using LLVM external calls.  

### Evaluation

In this section, we evalutate our implementation by using both unit tests and realistic benchmarks.

#### Unit Tests

The tests can be found [here](). Following we breifly illustrate what each test does by using the DFGs.

