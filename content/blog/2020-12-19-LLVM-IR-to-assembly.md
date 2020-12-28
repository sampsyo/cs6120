+++
title = "LLVM IR to Assembly"
[extra]
bio = """
   Socrates Wong is a second year PhD student at Cornell who interest lies in Computer Archicture.
"""
[[extra.authors]]
name = "Socrates Wong"
+++

# Modifying LLVM’s IR to assembly on a existing ISA with different number of registers. 
Most computer architectures have a fixed number of registers, and such number is often fixed and determined arbitrary.  Originally when 8086 was released (the original x86), there were 8 registers and most of them are specialized. However in most recent iterations of x86-64 it has total of 40, including 16 general purpose registers, 8 MMX registers, and 16 SSE registers.  As the registers area overhead in modern out of order processors (O3) are relatively small, it is often common practice to incorporate more registers than what is absolutely minimum to complete the task.   Many advanced topics in computing architecture such as simultaneous multithreading, Tomasulo algorithm, and speculative execution depend on the existence of having more physical registers than architectural registers.  
This begs the question of do we actually need the number of registers offered by the modern ISA? Or can we go back to the traditional AX, BX, CX, DX, SI, DI, BP, SP registers?  Although at the current landscape in O-3 processors the incorporation of registers are relative cheap, such relationships may not hold true in other alternative processing methods. (PIM, GPGPU, DNA computing, quantum computing, et cetera)  Therefore such dimensions are worth a closer inspection, and that we as researcher should not take the cheap overhead for registers in modern O3 processors for granted. 
This serves as the motivation of this project, in which I will modify LLVM assembly generation routine to generate code with less registers than what is offered by the traditional LLVM compiler.  Furthermore, an informal comparison on the performance tradeoff based on the number of architectural registers is done.  

# What did I do?
Although initially the project was to target the RISC-V ISA, but the current RISC-V Vector extensions currently still experimental and has an [outstanding pull request in Phabricator](https://reviews.llvm.org/rG5baef6353e8819e443d327f84edc9f2d1c8c0c9e) as of writing this blog post.  The lack of support in both LLVM side and the experimental status on the vector extension makes it hard for someone new to LLVM to effectly modify the praser in a resonable time, therefore I have decide to folow Adrian Sampson's advice in switching to x86 AVX and using that to benchmark. The plus side is that it is more mature and widely supported in the developer mailing list and the community.  However in retrospect it is something that I deeply regret as x86-64 is significently more complex.  This project is largely done by modifying the LLVM x86-64 target con figurations to create new target machine with different number of architectural registers.  Although X86-64 has better support across the LLVM stack, modifying the x86 machine target register configurations is no easy task and is significantly harder due to the number of aliasing it has across the different registers.   Most of the work that I have done on this project is modifying the LLVM assembly code generation to ulitizing different number of registers.  Then by executing the complied output from gcc, llvm and modified llvm.  I hope to unearth the relationship of architectural registers and performance.  To compare the results across multiple register types, I have done microbenchmark using Matrix Multiplication with three different x86-64 implementation (scalar, SSE, AVX-128bit, AVX-256bit) and benchmarking with FFT computations.  

# Hardest Part of the Project?
X86-64 is the hardest part of the project.  Although it is widely adopted, used and the codebase for x86 in LLVM relatively well maintained, the x86-64 ISA is so complex and riddled with special rules and cases that makes any simple modification very painful implement.  It is furthermore complicated by the complexity of the different concurrent calling conventions that x86-64 practices.  Going back to our recent class discussion about rebuild or fix, I think x86 is really something we need consider razing and rebuilding from starch.  It is not x86 is a bad ISA, but the current LLVM codebase in x86 is full of legacy code and code to emulate “bugs” (or features) in GCC compiler.  Although this dates to the history of computer, this makes x86-64 very unfriendly for any new developers to work on it.  Another notable mention is the amount of effort it takes to demystify the target machine files in LLVM.  

# Results and Success?
The project is a success in that I am able to report the performance compression with both microbenchmarks and on a FFT benchmark with varying number of architectural registers. 
On the microbenchmarks, I have choosen to compare the number of architectural registers of matrix multiplication in scalar execution, SEE, AVX4 and AVX8 on x86 machine.  For each one of the implementations, they are being compiled by a unmodified version of GCC, unmodified version of clang, and a modified version of clang with only 75% of the original x86-64 registers.    
The benchmark is primarily done by FFT and is implemented in SSE.   Thie code is compiled by three diffeent compliers: a unmodified version of GCC, unmodified version of clang, and a modified version of clang with only 75% of the original x86-64 registers.     
The benchmark is done on a 4 core i9-9900K CPU with 16 GB RAM in a Ubuntu VM.  The results are reported cycle count with rdtsc and execution time is reported as wall-clock time.  The results are verified by comparing the results of different implementations.


## Microbenchmark 

### Process and Methology
The microbenchmark preforms a 4x4 matrix multiplication.  A total of 4096 itnerations have been preformed, and the results are aggerate by the minimum function.  Due to the small amount of cycle it takes to compelte the opereation, using wall-clock time may introduced another source of error, therefore I have elected to report the results in CPU cycles.


### Results
Table 1, Microbenchmark of 4x4 matrix in CPU Cycles, 4096 trials, Minimum Number cycles required to complete the operation. 

| Microbenchmark 4x4   matrix             | gcc                   | clang  | clang (0.75) |
|-----------------------------------------|-----------------------|--------|--------------|
| REF                                     | 227.76                | 214.98 | 214.8        |
| SSE                                     | 144.06                | 124.48 | 124.33       |
| AV_4mem                                 | 140.11                | 170.03 | 170.14       |
| AVX_8                                   | 93.48                 | 98.99  | 95.75        |

### Results and Analysis
As this microbenchmark is designed to measure the maximum performance of the complied code, I have chosen to aggregate the result a min function. Preliminary analysis has shown that reducing the number of architectural registers has little or no effect on the performance on microbenchmarks.  This is as expected as microbenchmarks do not have high amounts register pressure as they often contain couple of register operands.  In this case, the complier backend implementation has more of an impact on the final performance then the modification on the number of x86-64 architectural registers.  


## FFT benchmark:

### Process and Methology
The benchmark preforms a set of FET calculations.  A total of 6 interactions have been performed and the data was preformed statistical analysis.  Execution time is measured in wall-clock time.

### Results
Table 2, FET benchmark in wall-clock time,average number cycles required to complete the operation.  
| Exec time (s) | gcc         | clang       | clang (0.75) |
|---------------|-------------|-------------|--------------|
| Average       | 2833314     | 2548542     | 3081897      |
| Min           | 2522625     | 2456454     | 2643296      |
| Std. Dev      | 381923      | 66116       | 497256       |
| CI-Lower(0.01)| 2431691     | 2479015     | 2558993      |
| CI-Upper(0.01)| 3234936     | 2618069     | 3604801      |
| CI-Lower(0.05)| 2527717     | 2495639     | 2684016      |
| CI-Upper(0.05)| 3138911     | 2601446     | 3479778      |

### Results and Analysis
In the FFT benchmark, the modified version with fewer registers consistently underperforms unmodified versions of GCC and Clang.  However, something of note worth is the minimum time needed for execution is relatively close for GCC and modified Clang version. At 0.01 significant the confidence interval overlaps between the categories clang and modified Clang , therefore we conclude the performance differences between group is not statistically significant. However, at 0.05 significance the results is statistically significant between original Clang and the modified Clang implementation.  In both cases, the performance difference between GCC and Clang are not statistically significant.  Despite the needed to have a greater sample size to have statically significant numbers for 0.01 significance level, and the need of a more comprehensive benchmark to identify the patterns and implications of reducing the number of architectural registers.  I have nevertheless demonstrated the ability to modify LLVM to generate assembly with a sub-set of all the original ISA registers.  

#Future Works?
There is couple of steps I will need to accomplish in the future.  The first and foremost is to abandon x86-64.  Although using x86-64 makes it a good academic exercise, it is nonetheless riddled with special exceptions and tons of specialized requirement to ensure support for legacy software.  This makes it very detrimental when attempting to isolate the performance gains and penalty.  The second point I would like to address is to use of a simulator to report the performance.   Compare to a live system with other workload, the simulator is both deterministic and is able to support multiple configurations to furthermore narrow down what really cause the performance decrease when the number of architectural registers decrease.  Lastly the incorporation of a comprehensive test bench to ensure that different workloads are fairly represented in the benchmarking.  



#Credits and Acknowledgements 
[LLVM](https://github.com/llvm-mirror/llvm/blob/master/CREDITS.TXT) and all its the contributors.  
[Fabian Giesen](https://github.com/rygorous) for his matrix multiplication benchmark.  
[Zhiics](https://github.com/zhiics)’s for his FFT benchmark
