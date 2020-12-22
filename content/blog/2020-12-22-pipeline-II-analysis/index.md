
+++
title = """Loop Pipeline Initiation Interval Estimation Using LLVM"""
[extra]
bio = """
[Jie Liu](https://github.com/Sibylau) is a 2nd year PhD student in Computer Systems Lab at Cornell. She is working on domain-specific language and accelerators.
"""
latex = true
[[extra.authors]]
name = "Jie Liu"
link = "https://github.com/Sibylau"
+++

## Introdution
Loop pipelining is an important performance optimization technique that exploits 
the parallelism among loop iterations. In unoptimized loops written in sequential languages 
like C/C++, one iteration can only begin after the previous iteration is complete. 
Loop pipelining allows loop iterations to overlap, which better utilizes resources and 
improves the overall throughput. To implement loop pipelining in hardware, however, it 
usually takes non-trivial efforts for hardware-description languages (HDL) programmers 
to produce a cycle-accurate register-transfer-level (RTL) design. Moreover, once the RTL 
design is fixed, it is difficult to retarget various design points because the datapath and 
control logics are explicitly constructed in the implementation.

With high-level synthesis (HLS) emerging as an alternative design fashion, programmers 
can simply add a pragma in a C/C++ loop to enable loop pipelining in hardware. However,
programmers at an early stage don't have much information on the performance of loops 
they write. If the specified pipeline initial interval (II) cannot be achieved, the HLS 
tools will throw out messages at least after running synthesis, which can take hours for 
a large design. In this project, I want to perform a quick analysis of the achievable loop 
pipeline II written in C-level. I believe early feedbacks will help programmers iterate 
their designs in a much faster way.

## Background
There are typically two factors of constraints that limit the degree of parallelism loop 
pipelining can exploit: one is hardware resource contention, and the other is the data 
dependencies between loop iterations. 

We denote the lower-bound II limited by resource usage confliction as ResMII, which is 
given by the formula (1):
![phase1](mii.PNG)

Where R is the set of available resources. $O_r$ is the number of operations in the loop 
body which occupy the resource $r$, and $N_r$ is the number of allocated resources, for 
example, the number of memory ports, or the number of DSPs. 

RecMII stands for Recurrence minimum II. It denotes the lower-bound of II due to loop-carried 
dependence. A loop-carried dependence indicates that operations in the subsequent loop 
iteration cannot start until the operations in the current iteration have finished. Array 
accesses are a common source of loop-carried dependence. For example, the following code 
snippet shows that the next iteration of the loop will read the array element updated by the 
current loop iteration. The minimum initiation interval, in this case, is the total number of 
clock cycles required for memory read, the add operation, and the memory write.

```c
for (i = 1; i < N; i++)
    mem[i] = mem[i-1] + i;
```

The overall minimum initiation interval, considering both resource constraints and data 
dependences, is $MII = Max(ResMII, RecMII)$, which takes the maximum achievable II under both 
circumstances. We can take a look at a warm-up example:

```c
int test()
{
  int A[20];
  for (unsigned t = 2; t <= 20 - 1; t++)
        A[t] = A[t - 2] + A[t - 1];
  return 0;
}
```

There are two load operations and one write operation in the loop body. If we assume the array 
`A` is mapped to a RAM with one read port and one write port, then we can calculate ResMII to 
be 2, as the number of memory read operations divided by the number of read ports. As for 
loop-carried dependency, we notice that the current loop iteration needs to read array elements
computed by the previous loop (`A[t - 1]`) and the one before the previous loop (`A[t - 2]`), 
which means the operations in the current loop cannot get scheduled until `A[t - 2]` and `A[t - 1]`
write back.  If we assume that the memory read and write operations both take 1 clock cycle, and 
the add operation, since it is a combinational logic, can be merged with memory read into one clock 
cycle, then the RecMII can be estimated as 2. The overall minimum initiation interval is $max(ResMII, 
RecMII) = 2$.

## Implementation
I implemented an [LLVM pass](https://github.com/Sibylau/LLVM_10.0_walkthru/tree/master/llvm-pass/loop_II)
to analyze the innermost loops. The problem is decomposed into two parts: estimate resource-constrained 
ResMII and data-dependency bounded RecMII.

For ResMII, it is easy to estimate using the formula (1). The pass traverses the loop, calculates 
the number of independent loads and stores in the code, and divide it by the number of available 
resources. I applied LLVM global value numbering pass on input programs before my handwritten pass, 
in order to avoid counting recurring memory accesses. The number of resources, as the denominator of 
the formula, is configured in advance as MACRO parameters. The pass I implemented only considered the
memory resource constraints. There are other types of resources that can incur usage conflict, for 
instance, the number of I/O ports, and certain compute resources, which I did not take into account 
for the course of simplicity.

The biggest challenge of this project is to estimate the RecMII, which basically requires precise 
loop-carried dependency analysis. Ideally, I can use the maximum dependency distance as an approximate 
of the RecMII. LLVM has a [loop dependency analysis pass](https://llvm.org/doxygen/DependenceAnalysis_8cpp_source.html). 
It successfully extracted the dependency information in simple test programs, such as the warm-up 
example above, but for some realistic benchmarks, it fails to figure out the dependence distance. 
In this case, to restore the estimated data-dependency bounded II, I computed the index difference of 
memory read-after-write (RAW) dependence pairs as a rough approximation. Since the actual RecMII depends 
on the number of cycles between memory read and write operations in the real schedule, the rough 
estimation performed in this pass can deviate from the real numbers. Additionally, the estimation is 
tuned towards a specific HLS tool, which is Catapult HLS tool in this project. I observed that if the 
loop body contains merely simple arithmetic operations on memory loads, and a single write-back to the 
memory, which is the case for all the tested benchmarks in PolyBench, the load-to-store operations 
scheduled in Catapult HLS tool are usually back-to-back and take 2 cycles. Therefore, by injecting 
into some tool-specific observation knowledge, the estimation of RecMII is able to be close to real 
numbers given by the tool. 

## Evaluation
I extracted 4 kernels in [PolyBench](http://web.cse.ohio-state.edu/~pouchet.2/software/polybench/)  that 
contain loop-carried dependencies. Memory accesses in PolyBench kernels are all affine accesses, which 
simplifies our analysis on the memory access index. The baseline numbers are given by running the key 
kernel code in Catapult HLS tool.


| Benchmarks  | Catapult HLS | My LLVM Pass | 
| ----        | -------------| ------------ |
| jacobi-1d   | ResMII = 3, RecMII = 1  | ResMII = 3, RecMII = 1  | 
| jacobi-2d   | ResMII = 5, RecMII = 1  | ResMII = 5, RecMII = 1  | 
| heat-3d     | ResMII = 7, RecMII = 1  | ResMII = 7, RecMII = 1  | 
| seidel-2d   | ResMII = 9, RecMII = 3  | ResMII = 9, RecMII = 2  | 


The experiment results show that the estimation of ResMII is precise for all test cases. The RecMII given
by our analysis pass is a good estimation for most benchmarks, except seidel-2d. Actually, for all benchmarks 
shown except seidel-2d, there is no real loop-carried dependency, and for this reason the pass returns 
RecMII=1. Seidel-2d has a memory RAW dependency, and the dependency distance cannot be identified by LLVM 
loop dependency analysis pass. Therefore, for this benchmark, the pass computes the index difference between 
the dependency load-store pair, which is 1, and adds the tool-specific additional schedule cycles to it, 
which outputs 2. It turns out that the estimation is over ideal, and the real II given by Catapult HLS tool is 3.

## Future work
 - For more precise estimation of ResMII, I need to add more types of resource constraints, e.g. interface ports, 
and compute resources.
 - For more precise estimation of RecMII, a more accurate and robust loop-carried dependency analysis is 
 required to attain the exact dependency distance. In order to tune the analysis pass towards an HLS tool, 
 I need to figure out a more systematic way to integrate scheduling information from the tool. The integration
 part can be flexible and portable which also embraces other simulation or synthesis tools. 
 [Gem5-SALAM](https://github.com/TeCSAR-UNCC/gem5-SALAM/blob/master/GEM5README) is a 
 recent work that leverages Gem5 hardware models to simulate system-scale accelerators. 

## References:
 -  [xilinx2015_2/sdsoc_doc manual](https://www.xilinx.com/support/documentation/sw_manuals/xilinx2015_2/sdsoc_doc/topics/calling-coding-guidelines/concept_pipelining_loop_unrolling.html#:~:text=A%20data%20dependence%20from%20an,called%20a%20loop%2Dcarried%20dependence.&text=In%20case%20of%20loop%20pipelining,operation%2C%20and%20the%20memory%20write)
 - Seto, Kenshu. "Scalar Replacement with Polyhedral Model." IPSJ Transactions on System LSI Design Methodology 11 (2018): 46-56.
