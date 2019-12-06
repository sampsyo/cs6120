+++
title = "A Spatial Path Scheduling Algorithm for EDGE Architectures"
extra.author = "Yi-Hsiang Lai"
extra.bio = """
  [Yi-Hsiang Lai](http://people.ece.cornell.edu/yl2666/) is a 4th-year Ph.D. student interested in computer architecture, high-level synthesis, and programming languages.
"""
latex = true

+++

This paper proposes a new scheduling algorithm for compiling programs to EDGE architectures, easpecially for the TRIPS architecture. EDGE (Explicit Data Graph Execution) is a class of ISAs different from the usual ISAs we have seen before (e.g., RISC and CSIC). It supports **direct instruction communication**. To be more specific, instead of specifying the source and destination registers of an instruction, direct instruction communication describes the producer-consumer behavior of a set of instructions. Namely, the instructions will be executed in a dataflow fashion. Once the input instructions are finished, the current instruction can fire. With the EDGE ISA, we need a corresponding compiler that schedules the instructions and maps them to the spatial architectures such as TRIPS. This paper first demonstrates how previous work tackles the problem. Then, the paper shows how to get the approximate optimal solutions with simulated annealing. Finally, it propses a new scheduling algorithm (i.e., spatial path scheduling (SPS) algorithm) and evaluates it with benchmarks from a wide range of applications. The results show that with the SPS algorithm, it can achieve substantial improvement compared with the previous work.

## TRIPS: An EDGE Architecture

With the rising complexity of existing applications and the need for low-power solutions for emerging silicon technology, we need an instruction-set architecture (ISA) that provides the following features. First, it can exploit different kinds of parallelism (e.g., data-level parallelism and thread-level parallelism) under fixed pipeline depth. Second, it can support power-efficient performance. Third, it can provide flexible data communication to avoid delays caused by long on-chip wires. Finally, it can run various applications with the same set of execution and memory units.

To achieve all four features, the TRIPS architecture builds up an array of execution units, where the instructions can be executed, and the computation results can be moved flexibly between different units. Following shows an example.

```
ADD r1 r2 r3
```

The above instruction is a traditional RISC instruction. It first reads values from register `r1` and `r2`. Then it adds the values then stores the result back to register `r3`. However, for an EDGE instruction, we do not specify the inputs, we only specify the outputs. Following is an example.

```
ADD T2 T3
```

In the above EDGE instruction, `T2` and `T3` are the destination execution units. Namely, after the addition, the result will be sent to the unit `T2` and `T3`. Each execution unit is implemented with an ALU and instruction buffers. In addition, the TRIPS architectures use **block-atomic execution**. Namely, the instructions are grouped into a block (usually consists of 128 instructions), which will be mapped to the execution units. The TRIPS architecture can file at most eight blocks at the same time. The architecture also has other components such as register files, caches, and control units. Since the architecture implements the EDGE ISA, each instruction describes which units the computation result will go. Similarly, an execution unit fires once all its inputs arrive.

## Compilation Flow for [TRIPS](https://ieeexplore.ieee.org/document/1310240)

A naive way to compile a program and deploy it to a TRIPS architecture is to generate the RISC assembly first. Then, the compiler analyzes the data dependence and builds a dataflow graph (DFG). The compiler also includes instructions from all branches of a conditional instruction (e.g., `br`). Next, according to the DFG, the compiler maps all instructions to the execution units with a scheduler. The compiler also needs to add data movement instructions if necessary. Finally, the compiler generates the EDGE ISA by referring to the mapping it creates in the previous stage. 

One observation from the compilation flow is that the out-of-order execution is enabled naturally with the TRIPS architecture. Namely, the execution order is determined dynamically at run time. The compiler only statically determines the instruction mapping. If we compare with VLIW, both the execution order and instruction mapping are determined statically at compile time. On the other hand, for an out-of-order pipelined processor, both the execution order and instruction mapping are determined at run time.

## Spatial Scheduling Problem

There is no doubt that there exist many optimization possibilities within this naive compilation flow. For instance, how should we group the instructions into blocks? How to perform branch prediction? And most importantly, how should we map the instructions to the execution units? The last problem is discussed in detail within the paper, where the authors first use simulated annealing to derive approximate optimal results. Then, they propose a spatial path scheduling (SPS) algorithm that maps the instructions to the execution units with several heuristics. Finally, the authors compare the SPS algorithm with a baseline algorithm proposed in 2004. 

The main idea of both the baseline and the SPS algorithms are the same. First, we create an initial set contains certain instructions. Then for each instruction in the set, we assign it with a **placement cost**. The instruction that has a higher cost will be scheduled first. After that, we add other instructions to the set until all instructions are scheduled. The difference between the baseline and the proposed algorithm is the function to calculate the cost.

### Greedy Algorithm - GRST

The initial set contains the instructions having no input or all its inputs are scheduled. For GRST, it adopts several heuristics. For example, it schedules the instructions within the critical paths first. It also considers data locality by placing load and store instructions next to the caches. Similarly, it places instructions that have register outputs next to the register files. Ideally, the distance between an instruction and the register is the same as the number of succeeding instructions until the final write operation. However, there exist many limitations. For example, the register files and caches are banked within the TRIPS architecture. This algorithm also serves as the baseline for the SPS algorithm that will be discussed later.

### Simulated Annealing 

One naive way to evaluate the scheduling results is by comparing them with the optimal results. However, it is an NP-complete problem. To solve that, the authors use simulated annealing, which can find an approximate optimal solution within a large search space. The cost function is defined as the number of cycles that are used to run the whole schedule. The authors further apply guided simulated annealing to gather the results with a shorter time.

### Spatial Path Scheduling (SPS)

The main idea of this algorithm is that, for each instruction, it calculates the cost for each possible position in the ALU array. Then the final placement cost is the minimal cost among all positions. Unlike GRST, the initial set contains the instructions having no or **at least one** scheduled input. Then after the calculation of the placement cost, the instruction with the highest placement cost will be scheduled first. This idea is similar to GRST, where the instructions in the critical paths will be scheduled first. To further improve the algorithm, the authors get the idea for other heuristics by comparing the results generated by SPS and simulated annealing.

#### Contention Modeling

The authors model two types of contention, which are ALU and link contention, respectively. For the first one, the idea is to solve the resource conflict when scheduling two instructions with the same ALU (i.e., execution unit). For ALU contention, we can further categorize it into two types. One is intra-block contention, and the other one is inter-block contention. For intra-block contention, it is similar to GRST, where the algorithm checks whether the two instructions that mapped to the same ALU may have resource conflict or not. If that is the case, then we add the penalty to the placement cost of the instruction we are scheduling. 

For inter-block contention, it is not handled by GRST. The idea is that the two instructions inside two different blocks may have resource conflicts. The authors try to add the number of consumers of the already scheduled instructions to the placement cost. However, this is too conservative since the consumers of the scheduled instructions do not necessarily have resource conflicts with the instruction that will be scheduled. To counter this, the algorithm removes candidates that must not have resource conflicts. For link contention, it is not trivial to calculate because the network utilization is unknown until run time. 

To calculate the final penalty, the algorithm sums up both intra- and intro-contention for ALUs. In addition, it introduces two factors, which are fullness and criticality, respectively. For the former, it corresponds to how full a block is. For the latter, it compares the maximum path length of an instruction with the length of the critical path. The final placement cost is just a combination of all the above numbers.

#### Global Register Prioritization

The main idea of this heuristic is performing loop-related optimizations. The algorithm performs the following three heuristics.
- Smaller loops (in terms of the number of blocks) will be scheduled first. 
- Loop-carried dependencies will be scheduled first. 
- When calculating the longest path, the algorithm will also consider the predecessor and the successor of the current block.

#### Path Volume Scheduling

The goal here is to find the best path from a source unit to a destination unit. This is not as simple as it seems because some units might be fully occupied by other instructions already. In addition, we need to fit all intermediate instructions to the path we find. The authors apply a depth-first search with **iterative deepening**. The idea is to set an upper bound first. It will keep increasing the upper bound until it finds a feasible solution.

#### Putting It All Together

Finally, we can combine all heuristics into a single function that calculates the placement cost. Namely, we first consider the penalty brought by the contention. Then, we consider the loop-related optimizations. Finally, we include the routing cost derived by the pathfinding algorithm.

## Evaluation

The authors selected a wide range of benchmarks with different levels of concurrency and memory behavior. Since dense blocks require more algorithmic work, the authors modified the selected benchmarks by hand. Then they compared the results with the baseline GRST algorithm and the simulated annealing results. They also show the results after applying each heuristic alone. In conclusion, their SPS algorithm improves the cycle count by over 21% in average comparing with GRST. Specifically, if we do not combine all heuristics, each heuristic can only provide at most 4% improvement in average. The path volumn scheduling cannot even provide improvement when being applied alone. Finally, if we compare with the simulated annealing results, the SPS results are within 5% difference in average. In addition, the authors performed a cross validation by performing the same algorithm on the unmodified benchmarks. The results still demonstrate an over 17% improvement in average comparing with GRST.

## Conclusion and Thoughts

This paper proposes a scheduling algorithm specific to the TRIPS architecture and it also shows decent evaluation results. This work is actually very interesting because there are more and more spatial architectures nowadays. For example, coarse-grained reconfigurable arrays (CGRAs) are very simimilar to this TRIPS architecture. The major difference is that CGRAs are reconfigurable. Namely, for TRIPS, each execution unit is just an ALU. On the other hand, for CGRAs, each compute unit is reconfigurable. They also share the same routing/scheduling problem. This problem is more complicated when we have other kinds of spatial architectures, such as FPGAs. If we make the problem more fine-grained, it becomes the real place & route problem during hardware synthesis.

## Questions
- What are the trade-offs brought by the TRIPS architecture compared with traditional multi-stage pipelined processors?
- How can we compare the proposed scheduling algorithm with other traditional algorithms, e.g., list scheduling, ASAP, ALAP? What are the differences?

