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

One key idea behind compiler optimization is to exploit the parallelism within a program. The most straightforward method is by optimizing loops. There exist many loop optimization techniques, including loop unrolling, reordering, tiling, pipelining, etc. Each technique can bring different effects to the given program. In this project, we focus on finding the pros and cons brought by loop unrolling with [Bril](https://github.com/sampsyo/bril) programs. We implement two methods for unrolling, which are static analysis and dynamic profiling, respectively. The source code can be found [here](https://github.com/seanlatias/bril/tree/arrays).

### Methodology 

#### Input Specification

In modern compiler frameworks, static loop unrolling passes usually follow a series of preprocessing passes. For example, 
LLVM has a loop canonicalization and simplification pass, which (1) makes sure that the loop has one and only one induction 
variable, (2) canonicalizes the loop induction variable to always start from zero and have unit stride, and (3) creates 
preheaders for loops that have multiple entrances. It is possible to further canonicalize the loop guard instruction and 
the condition used to check whether the loop bound is exceeded. To simplify the implementation of our static loop unrolling pass, without the loss of generality we make the following assumptions:
- The loop should have only one induction variable, and it can only be updated by addition or subtraction. The instruction
that updates the induction variable should directly store to the induction variable. 
- When generating the condition variable to check the loop bound, a ``lt`` instruction must be used and the loop induction variable must be its first argument. 
- The loop guard instruction must be a ``br`` instruction, and it always jumps out of the loop when the condition is ``False``. 
- We further assume a copy propagation pass has been performed on the code, i.e. we won't track the copy chain to find where the induction variable is actually updated. 

#### Loop Identification

The first step is to identify loops from a Bril program. In other words, we need to find **back-edges** within a control-flow graph. A back-edge is defined as a directed edge pointing from basic block A to basic block B where B dominates A. [(Additional readings on basic blocks and finding loops)](https://www.csl.cornell.edu/courses/ece5775/pdf/lecture06.pdf)

#### Static Analysis
The first part of this project is to implement a completely static compiler pass to do loop unrolling for Bril programs. 
Our pass performs complete unrolling only. 
This pass calls several passes that are already present in the Bril repo. In addition, we implemented a reaching definition
pass using the code template in ``examples/df.py``, updated the constant propagation pass to make it more powerful,
and slightly changed the cfg construction pass  in ``examples/cfg.py`` so that each basic block has a label. The overall 
flow of the pass is as follows:
- Construct the CFG, then compute predecessors, successors, and dominators of each basic block. 
- Run the constant propagation and reaching definition passes. 
- Use CFG information to find loops in the program. Notice that we only unroll loops with regular structures. Our definition
of "regular structure" is given in the implementation detail section. 
- For each loop that can be unrolled, leverage the information from constant propagation and reaching definition passes to calculate the tripcount. 
- If the tripcount is computable, decide whether we want to unroll the loop or not using a simple criterion: the total 
number of instructions in the loop (tripcount * number of instructions in all loop basic blocks). Currently we unroll loops
with less than 1k instructions. 
- Unroll the loop by first duplicating all basic blocks inside the loop and then removing redundant control flow instructions. 

#### Dynamic Profiling

The key idea of dynamic profiling is to estimate the loop status by simulating the loop execution. Since the loop trip count may depend on different conditions, it will hard to infer the loop bound without the information from actual execution environment. For example, induction variable being udpated indirectly can make the unrolling problem complicated. An intuitive way of simplifying the problem is to proflie the loops before unrolling. Here we extend the data flow analysis pass to support dynamic profiling, i.e. we maintain a status table to track the update history for each variables and . By refering to the table and locate the update history of the back-edge related instructions and basic blocks, we can easily infer various information needed, e.g. trip count, induction variable as well as loop stride. Here to simplify the analysis, we assume all loops to be unrolled are natural loops, namely the program can only enter the loop from the header. Under the assumption that the loop bound is invariant in multiple runs, the basic blocks in the loop body will be repliacted for trip bound times. Back edges (i.e. jmps) and condition check statements will be removed form originla bril program.

### Implementation details

#### External Packages
In this project, we use two external packages. The first one is the array extension of Bril, whose source code can be found [here](https://github.com/Checkmate50/bril/tree/arrays). The second one is the C code generation for Bril. The source code can be found [here](https://github.com/xu3kev/bril2c).

#### Static Analysis
This section details how we implement the static loop unrolling pass. We first state our definition of "regular loops" that
can be unrolled by our pass. A "regular loop" should have the following properties other than the ones mentioned in the 
input specification section:
- The control flow can only leave the loop from either of the two blocks connected by the back-edge. 
- The loop can not contain any sub-loop. 

In this section, without special mentioning, we will use *entry* to denote the entrance block of the loop (destination of 
the backedge), and use *exit* to denote the exit block of the loop (source of the backedge). The implementation details of 
each step outlined in the Methodology section are show below. 

##### CFG Analysis
We use the functions provided in ``examples/cfg.py`` to construct the CFG and compute the predecessors and successors of 
each basic block. The ``block_map`` function is slightly modified to give each basic block a label, inserted as the first
instruction of the block. We use the pass provided in ``examples/dom.py`` to compute the dominators of each basic block. 

##### Dataflow Analysis
We implemented a reaching definition pass in ``examples/df.py``, which is useful in finding the last instruction in the loop
that updates the induction variable and the condition variable. Our implementation conforms with the pseudo-code in the 
lecture notes. We also augmented the constant propagation pass so that it can propagate through ``id`` and arithmetic 
instructions. The results of the constant propagation pass is used to find the initial value of the induction variable, 
the stride of the induction variable, and the loop bound. 

##### Finding Loops
Our pass detects loops by first finding a backedge and then adding nodes into the loop. A back-edge is defined as a directed
edge pointing from basic block A to basic block B where B dominates A. After locating the back-edge, we add all nodes that
can reach the *exit* and are dominated by *entry* into the loop. We also filter out loops that are not regular by our standard and loops with sub-loops in this step. 

##### Computing Tripcount
To compute the tripcount of the loop, we need to obtain the initial value, stride, and final value of the induction 
variable. If these values are constants, they can be computed by the constant propagation pass. The issue is how to properly
locate two instructions: the instruction that updates the condition, and the instruction that updates the induction variable. 

We can easily locate the branch instruction that can direct the control flow out of the loop. We are interested in the condition variable used for this branch. Here we use the results of reaching definitions to find all possible reaching 
definitions of this variable from blocks inside the loop. If there are more than one reaching definition, the loop tripcount
can not be easily computed and we conservatively choose to not unroll the loop. Otherwise, we will be able to locate the 
instruction used to generate this condition. According to our assumption on the program, the second argument of this instruction is the loop bound. If the value of this argument can be found in the constant propagation result, then we have 
determined the loop bound. Otherwise, the tripcount can not be computed and we choose not to unroll the loop. 

The first argument of the condition instruction is the induction variable. We further find its reaching definition from 
the loop in the current block. If there is only one reaching definition, then we check whether this target instruction 
updates the induction variable in the way we expected. If so, the stride of the loop can be determined by checking the 
constant in the target instruction. If the stride can not be retrieved then we don't unroll the loop. 

To obtain the initial value of the induction variable, we check the result of constant propagation in the *entry* block. 
Notice that we only care about the **out** sets of basic blocks that are (1) not in the loop, and (2) can reach the *entry* block. If we can obtain one single constant value of the induction variable by examining all these out sets, then we have 
successfully determined the initial value of the induction variable. Otherwise, the loop won't be unrolled. 

Now the tripcount can be computed by ``(bound - initial_value + stride - 1) // stride``. Notice that with this formula, a 
negative tripcount refers to an infinite loop. 

##### Unrolling the Loop
We can finally unroll the loop now. Since we have inserted a label for each basic block, we can simply duplicate the basic
blocks inside the loop and redirect the control flow when it reaches the backedge. If the control flow can exit the loop from the *entry*, we need to duplicate the *entry* block one more time. All but the last duplicates of the ``br`` instruction for the back-edge can be safely converted to ``jmp`` instructions. 

The generated code would be functionally correct. 
However, it is not very efficient because of the redundant ``br`` and ``jmp`` instructions: we already know where the control flow will go to at the backedge. Therefore, it is ideal to merge the basic blocks of the unrolled loop as much as 
possible. We use a trick to achieve this: rather than actually merging the basic blocks in the code, we leverage the orderedness of the ``OrderedDict`` and ``list`` data structures in Python to force a meaningful order of basic block 
placement. What we actually do is to remove the labels of all but the first duplicate of the *entry* block and all the *exit* blocks, and remove all the ``jmp`` instructions of all the intermediate duplicates of the *entry* and *exit* blocks. 

#### Dynamic Profiling 

The dynamic profiling based unrolling pass consists of two parts mainly: loop status analysis, program structure re-construction. In the loop status analysis part, we created a variable table to track the defition and usage for each varible. The table is used as a look-up table for loop analysis and basic block replication in later stages. The variable table can be extended to record the data dependency between different varibles, and this information can be used to realize loop normalization, copy propagation and such. With the update information for each variables, we then iterate all the back edges in the control flow graph and unroll the loops accordingly. Note that the back edges are stored in a dictionary and need to be processed in order to make sure the loops can be unrolled correctly (e.g. for nested loops, the inner most loop needs to be unrolled first). In the last step, we replicate the loop body and remove the condition branch instructions as well as the back edges to construct the new unrolled program.

### Experiment Results

We use the hand-written test programs to verify the correctness of our loop unrolling passes and measure performance improvements. The functioanl correctness of the unrolled code can be verified with bril intepreter. To evaluate the performance between original and unrolled programs, we converted the bril code from json representation to C using the Bril C back end. We use GCC to compile the C program without any optimization (``-O0``), run it for 10000 times with bash loop to measure the total time consumption. Experiments are performed on a server with an 2.20GHz Intel Xeon processor and 128GB memory. All programs are single-thread. The benchmarks can be found [here](https://github.com/seanlatias/bril/tree/arrays/test/unroll).

#### Static Analysis Evaluation
The program is located [here](https://github.com/seanlatias/bril/blob/arrays/examples/loop_unroll.py). To execute the pass, run the following command.

``python loop_unroll.py output.json < input.json``

The pass can successfully unroll the loops in our test programs. For the "double loop" test case, the pass correctly identifies the inner loop and unrolls it. Execution time of the original and unrolled versions are as follows:

| Benchmark | Original (s) | Unrolled (s) |
|:---------:|:------------:|:------------:|
|Two consecutive loops|3.530|3.621|
|Nested loop|3.509|3.533|
|Vector sum|3.599|3.508|
|Set vector value|3.473|3.671|

There is no significant performance difference between the unrolled version and the original version. One reason might be 
because the loop tripcounts in the test programs are all very small. As a result, we modify the vector sum program to have 
large tripcount (2000) and compare the performance again. To enable this we modify our loop unrolling pass to allow unrolling very large loops. In terms of the sizes of the executables, the unrolled version is 12.7x larger than the original version. The original and unrolled program uses 3.693s and 3.755s, repectively. There is still no significant performance benefit. The possible reason is that the control flow of our test examples is too regular and can be perfectly predicted by the branch predictor in modern processors. Since the processor is pipelined, we do not observe significant performance improvement even after we remove the redundant branches after unrolling. For most benchmarks we actually observe slowdowns, which possibly result from instruction cache misses caused by larger executable sizes. 

#### Dynamic Profiling Evaluation
The program is located [here](https://github.com/seanlatias/bril/blob/arrays/examples/dyn.py). To execute the pass, run the following command.

``python dyn.py < input.json > output.json``

The pass can run successfully in different test cases including consecutive loops and double loops. In this section we mainly focus on the performance perspective. We construct a vector add program with large trip count numbers (2000), which can benefit loop unrolling and out-of-order execution of modern processors. The original and unrolled program takes 9.111s and 8.341s (measured with linux time utility in usr time) for 10000 execution.

### Other Features
We also improve the constant propagation method provided. With this improvement, we can now perform arithmetic simplification. This pass can be applied after loop unrolling, which can potentially improve the performance by reducing the number of arithmetic operations.
