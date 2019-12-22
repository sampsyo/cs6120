> cat index.md                                                Sat Dec 21 17:53:42 2019
+++
title = "Register Allocation for Bril"
extra.author = "Hongbo Zhang, Sachille Atapattu, Wen-Ding Li"
extra.bio = """
  Hongbo Zhang is a first-year Ph.D. student in computer science. He is interested in systems and computer architecture. He is also an okay archer shooting recurve bow.
  [Wen-Ding Li](https://www.cs.cornell.edu/~wdli/) is a Ph.D. student at Cornell interested in high-performance code generation.
  Scachille Atapattu is a Ph.D. student at Cornell University.
"""
+++
# Register Allocation for Bril

Processors usually have registers that can be used to store variables for faster access. 
Programs will run faster if we can put variables in these registers. 
Nevertheless, registers are limited, so oftentimes we still need to access memory (or cache) instead of registers. 
Memory access is costly compared with register access. 
Although modern programs often use a large number of variables, 
we can analyze their live ranges so the same registers may be reused to hold different variables throughout the program. 
Then, we can put more variables in the registers which may lead to fewer memory accesses and yield a faster program. 

The goal of our project is to perform register allocation on the Bril program and analyze its performance. 
In this project, we implemented register allocation via graph coloring. 
Our project can be found in [this repository](https://github.com/xu3kev/bril/regalloc).

## Method
Our approach for register allocation can be divided into three main parts. 
First, we perform a liveness analysis and then build a graph. 
Second, we solve the approximate k graph coloring problem to obtain a mapping from each variable to the corresponding register. 
At last, we generate Bril code based on the mapping.

### Liveness Analysis
We use data flow analysis to get the live variables at each instruction of the program. 
First, we transform the input Bril program to the control flow graph (CFG) and then perform liveness analysis with standard backward flow analysis. Once we obtain the live variables at the end of each code block, we start from that and go backward: when we see a definition of a variable, we kill the variable, and when we see the use of the variable, we mark it as alive. Then, we can get the live variables at every instruction.

### Graph Coloring
With the liveness ranges of all local variables, 
we can know which variables are "alive" at the same time. 
If two variables have overlapped liveness ranges, 
it means that they cannot be allocated to the same register. 
It is not always possible to find a register allocation for variables.
For example, it is not possible to allocate 2 registers to 3 variables with overlapped liveness ranges. 
Our goal is to have a register assignment so that 
we can allocate as many variables on registers as possible.

If we create a graph such that each node represents a local variable.
There is an edge between two nodes if the two variables have overlapped the liveness ranges. 
Now solving register allocation is equivalent to finding the largest $k$-colorable subgraph,
which does not have a polynomial-time algorithm to solve it.

Instead of finding the optimal coloring, we use optimistic coloring to find one large subgraph with as many nodes as possible.
On each iteration, we find the node with the least number of neighbors that are not assigned a color yet. Then we try to assign a color to the selected node based on the constraints of all neighbor nodes. If it is not possible to assign a color to the selected node, then we remove the node from the graph along with all edges connected to the node.
Only the variables with an assigned color will be allocated to the corresponding registers, all reads or writes to other variables will involve a memory load and store.

### Code Generation
Once we have the graph coloring scheme, we start to generate a new Bril program with register allocation.
We use special variables `r_xx` to represent registers.  For example, 
```
a: int = const 5;
b: int = const 3;
c: int = add a b;
print c;
```
with `r_01` allocated to `a` and `r_02` allocated to `c` will look like the following:
```
r_01: int = const 5;
b: int = const 3;
r_02: int = add r_01 b;
print r_02;
```

If any data transferring between registers and stack is needed, it could be done by `id` operations like `a: int = id r_01`.

## Evaluation Method
We evaluate our implementation by counting load and store counts during program execution.

### Counting the Number of Memory Access
We modified the Bril interpreter `brili` to count the number of implicit memory access operations for those data not in registers. As discussed in the code generation section, we introduce a special rule, that is, variable names prefix with `r` are registers. Read and write operations on those registers do not require memory access. However, read and write on other variables all are counted as memory access. With this rule, we can do register allocation and evaluate on Bril without language breaking changes. For example:

- `a: int = const 2;` has 1 memory access operations (store a)
- `a: int = id b;` has 2 memory access operations (load b, store a)
- `c:int = add a b;` has 3 memory access operations (load a, load b, store c)
- `br given a b;` has 1 memory access operation (load given)
- `print <N args>;` has N memory access operations, N being the number of arguments (N loads)

As for accessing register values (the case where special variables represented registers will not be counted as a memory access):

- `r1: int = const 2;` has 0 memory access operations
- `a: int = id r2;` has 1 memory access operations (store a)
- `c:int = add a r2;` has 2 memory access operations (load a, store c)

We compare the total number of memory access operations for the same program with and without register allocation. 

It was identified during the evaluation that this implementation of counting doesn't support the `ret` instruction. We hope to fix this issue shortly. We have tested this interpreter version with some handwritten register allocated code.

### Benchmarks
We wrote some common kernels and some handwritten programs that can be mapped with a specific number of registers to test our register allocation performance. The following table provides a list of these benchmarks.

| Test   | Description |
|--------|---------------------------------------------------------------------------------------|
| br     | Tests with branching                                                          |
| if     | Tests with a more complex conditional                                        |
| loop   | Tests with a loop                                                          |
| alloc1 | Tests a simple 4 variable program taken from [these](https://web.stanford.edu/class/archive/cs/cs143/cs143.1128/lectures/17/Slides17.pdf) slides.    |
| alloc2 | Test with 14 variables possible to allocate to 4 registers, inspired from the [book](https://www.google.com/books/edition/Modern_Compiler_Implementation_in_Java/N-sgAwAAQBAJ?hl=en&gbpv=0) |
| alloc3 | Test with 14 variables possible to allocate to 6 registers, adopted from the [book](https://www.google.com/books/edition/Modern_Compiler_Implementation_in_Java/N-sgAwAAQBAJ?hl=en&gbpv=0) |
| matmul | An implementation of matrix multiplication in Bril                                |
| dotprod | An implementation of dot product in Bril                                |
| fib | An implementation of Fibonacci in Bril                                |
| fact | An implementation of factorial in Bril                                |
| polymul | An implementation of polynomial multiplication in Bril                    |

The last 5 benchmarks are from this [repository](https://github.com/xu3kev/bril/tree/master/benchmark).

### Baseline
To evaluate our register allocation, we assume that there is no register holding data across different instruction as the baseline so that each instruction loads all operands from the memory and stores the result back to memory after execution. This is a conservative approach and provides an upper bound to the number of loads and stores that can occur, so register allocation should have less or the same number of load/store operations.

### Simulated load and store counts

We simulated the baseline and graph coloring implementation using Brili to test for two things,
1. To make sure that the functionality hasn't been affected by our optimization
2. To measure memory access operations as a proxy for performance

The following tables provide the memory access count in the form of `loads/stores`. These tests are run assuming for 4 registers.

| Test    | baseline  | graph coloring |
|---------|-----------|----------------|
| br      | 3/4       | 0/0            |
| if      | 14/11     | 1/1            |
| alloc1  | 6/5       | 0/0            |
| alloc2  | 20/17     | 4/3            |
| alloc3  | 20/16     | 8/4            |
|---------|-----------|----------------|
| loop    | 53/24     | 21/2           |
| matmul  | 2048/1216 | 1008/126       |
| dotprod | 257/145   | 169/78         |
| fib     | 103/56    | 32/12          |
| polymul | 929/509   | 869/505        |

The figure illustrates how effective graph coloring is by the percentage reduction in memory accesses. Apart from polymul it shows a significant reduction in loads and stores. A possible reason for poor performance in polymul might be its high variable usage. 
<br>
<img src="./reg_red.png" style="width: 80%">
<br>

| Test   | baseline | 1 register | 2 registers | 2 registers | 8 registers |
|--------|----------|------------|-------------|-------------|-------------|
| br     | 3/4      | 1/2        | 0/1         | 0/0         | 0/0         |
| if     | 14/11    | 9/6        | 5/3         | 1/1         | 0/0         |
| alloc1 | 6/5      | 3/3        | 0/0         | 0/0         | 0/0         |
| alloc2 | 20/17    | 14/11      | 10/8        | 4/3         | 0/0         |
| alloc3 | 20/16    | 15/11      | 12/8        | 8/4         | 0/0         |

We have also tested the graph coloring implementation by sweeping the number of registers. As expected, memory accesses go down as the number of registers increase. 

