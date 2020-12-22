+++
title = "LLVM IR to Assembly"
[extra]
bio = """
   Socrates Wong is a second year PhD student at Cornell who interest lies in Computer Archicture..
"""
[[extra.authors]]
name = "Socrates Wong"
+++

## Modifying LLVM’s IR to assembly on a existing ISA with different number of registers. 
Most computer architectures have a fixed number of registers, and such number is often fixed and determined arbitrary.  Originally when 8086 was released (The original x86), there was only 8 registers and most of them are specialized, however in most recent iterations of x86-64 it has total of 40, including 16 general purpose registers, 8 mmx registers, and 16 SSE registers.  As the registers area overhead in modern out of order processors(O3) are relatively small, it is often common practice to incorporate more registers then what is absolutely minimum to complete the task.   Many advances topics in computing architecture such as simultaneous multithreading, Tomasulo algorithm, and speculative execution depend on the existence of having more physical registers then architectural registers.  
This begs the question of do we actually need the number of registers offered by the modern ISA? Or can we go back to the traditional AX, BX, CX, DX, SI, DI, BP, SP registers?  Although at the current landscape in O-3 processors the incorporation of registers are relative cheap, such relationships may not hold true in other alternative processing methods.  Therefore such dimensions are worth a closer inspection, and that we as researcher should not take the cheap overhead for registers in modern O3 processors for granted. 
This serves as the motivation of this project, in which I will modify LLVM assembly generation routine to generate code with less registers than what is offered by the traditional LLVM complier.  Furthermore, an informal comparison on the performance tradeoff based on the number of architectural registers is done.  

## What did I do?
Although initially the project wants to target the RISC-V ISA, but the current RISC-V Vector extensions currently still experimental and have outstanding pull request in phabricator as of writing this blog post. [1](https://reviews.llvm.org/rG5baef6353e8819e443d327f84edc9f2d1c8c0c9e) The lack of support in both LLVM side and the experimental status on the vector extension makes it hard for someone new to LLVM to effectly modify the praser in a resonable time, therefore I have decide to folow Adrian Sampson's advice in switching to x86 AVX and using that to benchmark. The plus side is that it is more mature and widely supported in the developer mailing list and the community, however in retrospect it is something that I deeply regret as x86-64 is significently more complex.  This project is largely done by modifying the LLVM x86-64 target configurations to create new target machine with different number of architectural registers.  Although X86-64 has better support across the LLVM stack, modifying the x86 machine target register configurations is no easy task and is significantly harder due to the number of aliasing it has across the different registers.   Most of the work that I have done on this project is modifying the LLVM assembly code generation to ulitizing different number of registers.  Then by executing the complied output from gcc, llvm and modified llvm.  I hope to unearth the relationship of architectural registers and performance.  To compare the results across multiple register types, I have done microbenchmark using Matrix Multiplication with three different x86-64 implementation (scalar, SSE, AVX-128bit, AVX-256bit) and benchmarking with FFT computations.  
Hardest Part of the Project?
X86-64 is the hardest part of the project.  Although it is widely adopted, used and relatively well maintained. The x86-64 ISA is so complex and riddled with special rules and cases that makes any simple modification very painful implement.  It is furthermore complicated by the complexity of the different concurrent calling convention that x86-64 practices.  Going back to our recent class discussion about rebuild or fix, I think x86 is really something we need consider razing and rebuilding from starch.  It is not x86 is bad ISA, but the current llvm code base in x86 is full of legacy code and code to emulate “bugs” (or features) in GCC complier.  Although this dates to the history of computer, this makes x86-64 makes it very unfriendly for any new developers to work on it.  Another notable mention is the amount of effort it takes to demystify the target machine files in LLVM.  

## Results and Success?
The project is a success in that I am able to report the performance compression with both microbenchmarks and on a FFT benchmark with varying number of architecture registers. 
On the microbenchmarks, I have choose to compare the number of architecture registers of matrix multiplication in scalar execution, SEE, AVX4 and AVX8 on x86 machine.  For each one of the implementations, they are being compiled by a unmodified version of GCC, unmodified version of clang, and a modified version of clang with only 75% of the original x86-64 registers.    
The benchmark is primarily done by FFT, which is implement in SSE.   Which is also being complied by a unmodified version of GCC, unmodified version of clang, and a modified version of clang with only 75% of the original x86-64 registers.     
The benchmark is done on a 4 core i9-9900K CPU with 16 GB ram in a ubuntu VM.  The results are reported cycle count with rdtsc and execute time is reported as wall time.  The results are verified by comparing the results of different implementations.
The results are outlined as below:

## Microbenchmark:

| Microbenchmark 4x4   matrix (min cycle) | gcc                   | clang  | clang (0.75) |
|-----------------------------------------|-----------------------|--------|--------------|
| REF                                     | 227.76                | 214.98 | 214.8        |
| SSE                                     | 144.06                | 124.48 | 124.33       |
| AV_4mem                                 | 140.11                | 170.03 | 170.14       |
| AVX_8                                   | 93.48                 | 98.99  | 95.75        |

## FFT benchmark:

| Exec time| gcc         | clang       | clang (0.75) |
|----------|-------------|-------------|--------------|
| Average  | 2833314.167 | 2548542.667 | 3081898      |
| Min      | 2522625     | 2456454     | 2643296      |

Preliminary analysis has shown that reducing the number of architectural registers has little or no effect on the performance on microbenchmarks.  This is as expected as microbenchmark usually do not have much register pressure as they often contain couple of operands.  In the FFT benchmark, our modified version with less registers consistently underperforms unmodified versions of gcc and clang.  However, something of note worth is the minimum time needed for execution is relatively close for gcc and modified clang version.  Although the data is still not statistically significant due to the small amount of sample size and the noise introduced my running code in non-dedicated hardware, it shows that the number of architectural registers has a greater impact on the average execution time then minimum execute time.   
Despite needing a greater sample size to have statically significant numbers, and the need of a more comprehensive benchmark to identify the patterns and implications of reducing the number of architectural registers.  I have nevertheless demonstrated the ability to modify LLVM to generate assembly with a sub-set of all the original ISA registers.  

#Future Works?
There is couple of steps I will need to accomplish.  The first and foremost is to abandon x86-64.  Although using x86-64 makes it a good academic exercise, it is nonetheless riddled with special exceptions and tons of specialized requirement to ensure support for legacy software.  This makes it very detrimental when attempting to isolate the performance gains and penalty.  The second point I would like to address is to use of a simulator to report the performance.   Compare to a live system with other workload, the simulator is both deterministic and is able to support multiple configurations to furthermore narrow down what really cause the performance decrease when the number of architectural registers decrease.  Lastly the incorporation of a comprehensive test bench to ensure that different workloads are fairly represented in the benchmarking.  

#Credits and Acknowledgements 
[LLVM](https://github.com/llvm-mirror/llvm/blob/master/CREDITS.TXT) and all its the contributors.  
[Fabian Giesen](https://github.com/rygorous) for his matrix multiplication benchmark.  
[Zhiics](https://github.com/zhiics)’s for his FFT benchmark
