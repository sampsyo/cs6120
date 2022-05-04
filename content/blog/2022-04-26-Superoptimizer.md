+++
title = "Superoptimizer: a look at the smallest program"
[[extra.authors]]
name = "Jonathan Tran"
[[extra.authors]]
name = "Victor Giannakouris"
+++

## Introduction

The majority of compilers utilize optimization to improve code, in terms of runtime performance, code size, and other factors. However, optimization does not necessarily generate the optimal program; the optimization process merely generates a better candidate program. In the Superoptimization paper by Massalin,, a different paradigm is introduced: optimization is performed to generate the minimal sized program, producing programs that are smaller compared to traditionally compiler-optimizer programs, as well as human hand-optimized programs [^1].

As an illustration of Superoptimizer’s capabilities, consider an example centered on compiling the signum function, which is defined as follows: 

```int signum(x) {
	If (x > 0) return 1;
	Else if (x <  0) return -1;
	Else return 0;
}
```

The author notes this program usually compiles to around 9 instructions, including several conditional jumps, while a human can manually optimize this program to 6 instructions. Remarkably, Superoptimizer finds an even smaller solution, consisting of 4 instructions, which is shown below.

```
Add.l d0, d0
Subx.l d1, d1
Negx.l d0
Addx.l d1, d1
```

This solution is guaranteed to use the least number of instructions to calculate the signum function, and avoids the use of any conditional jumps. Assuming this function is used many times, the impact of the Superoptimizer is tremendous, by saving 50% of instructions from a non-optimized solution, and also improving on pipelined execution, due the elimination of jumps. Furthermore, we know that in the above architecture, there is no shorter instruction solution, so there is no need for further optimization. As a result of this example, we can see the power of using the Superoptimizer.  

The Superoptimizer can be generalized to be used in a variety of circumstances, ranging from commonly used mathematical routines, to chunks of code that have high usage. There are several examples presented in the paper ranging from architectural design to generating pruning sequences for the superoptimizer itself. 

Finally, the ideas introduced by the Superoptimizer are used in various other works today, and continue to influence research in compiler optimizations. The brute force search method used in the Superoptimizer has been refined in various ways, ranging from rule-guided search in the Denali compiler, to stochastic search used by STOKE [^3][^4]. The optimality of Superoptimizer’s generated code has been used to create “libraries” of peephole optimizations that can be applied by real world compilers. All of these works build upon the ideas introduced by the Superoptimizer to address deficiencies and build even better compilers. 

## Core Ideas

The core idea of the Superoptimizer paper is to use brute-force enumeration to find the smallest-sized program `P’` that is semantically equivalent to the original program `P`. The enumeration process begins by considering all programs with 1 instruction, and checks if the program is equivalent to the original program. If none of the programs of size 1 are equivalent to the original program, then programs composed of 2 instructions are considered. This process continues until an enumerated program is found to be semantically equivalent to the original program. By following this procedure, it is guaranteed that the discovered program will be the smallest program in size with the same behavior as the original program. 

One problem arising from brute force enumeration is the excessive runtime needed to generate programs. Massalin proposes a pruning method to eliminate programs that are clearly not minimal in size. The Superoptimizer keeps track of redundant sequences of instructions, or sequences that do not have any effect on the program. Whenever an enumerated program contains these sequences, the program is not considered, because it is not minimal. 

For example, an instance of pruning occurs with the series of instructions `move x y; move x y`. The first instruction is identical to the second, and so the combined effect is equivalent to just a single move instruction. When pruning is used, this redundancy, and others like it, are eliminated from program enumeration. 

Brute force enumeration generates programs, but not all of these programs have the same semantic behavior as the original program. One way to ensure that the new enumerated program acts like the original program is to check it on every input. To do this, one can construct boolean minterms representing the original and the proposed program, and compare the minterms for equivalence. However, this technique fails to scale in terms of performance. In particular, the author notes that such a method only allows enumerations of programs with up to 3 instructions. Additionally, such a technique cannot apply to instructions that use pointers. Checking equivalence with pointers causes an doubly-exponential increase in the number of minterms that makes it infeasible to even compare minterms for limited memory machines. 

Massalin uses a clever technique to solve this problem. Since enumerated programs are likely to differ on many inputs from the original program, to eliminate semantically different programs, it suffices to create small test vectors as a filtering mechanism. Test vectors are manually chosen, and Massalin used test vectors that included edge cases for the type of program that was being tested, as well as more-general inputs ranging from -1024 to 1024. 

While test vectors are likely to eliminate many semantically different enumerated programs, some enumerated programs may still fall through the cracks. At this point, it is the responsibility of the operator of the Superoptimizer to manually ensure that the proposed program is equivalent to the original program. 

## Applications

The author proposes several scenarios where Superoptimizer can be applied. The simplest scenario is to use the Superoptimizer on small chunks of code, such as small functions. For instance, the author applies the Superoptimizer on small mathematical functions such as the absolute value function or maximum and minimum functions. On the assumption that these functions are heavily used, the application of Superoptimizer on these functions can bring great improvement. 

The Superoptimizer can also be applied to larger functions, with the caveat that some clever work has to be done to allow the Superoptimizer to run in a feasible time frame. For instance, the author applied the Superoptimizer to optimize a `printf` function.  By applying the Superoptimizer to small chunks of the `printf` function, the function could be improved incrementally. In fact, using the prior approach seems even better when one has an idea of what types of code are more frequently used. Having a profiler guide where these small chunks of code can be superoptimized seems like one approach to use this technique. 

Another way the Superoptimizer can be used is to identify sequences of instructions that can be used for pruning. Here, the Superoptimizer can enumerate sets of instruction sequences equivalent to a single instruction. When a sequence of instructions is found to be in that set, we know that it is redundant to the original single instruction, and thus can be pruned. 

Finally, the author also gives an instance of using the Superoptimizer to discover equivalent statements in computer architecture. As some instruction set architectures contain many equivalent instructions, it is possible to use the Superoptimizer to search for an equivalent sequence of simpler instructions that can make up a more complex instruction in the ISA. 

## Related Works

The Superoptimizer has several limitations which influence how the Superoptimizer can be applied. One of the major limitations of Massalin’s work is that Superoptimizer takes a long time to run and find optimal programs. Another downside is that Superoptimizer only works on the Motorola 68000 and Intel 8086 architectures. Finally, yet another issue with Superoptimizer is that enumerated programs are not guaranteed to be equivalent to the original program. These downsides influence the design of future projects that build upon the original Superoptimizer. 

For instance, the GNU Superoptimizer project made the Superoptimizer project portable across many different architectures [^2]. The GNU Superoptimizer also made superoptimization to the commonly used GCC compiler, allowing this concept to be applied to real world programs. This work ultimately expands the scope of the Superoptimizer across many architectures, but it still suffers from the same limitations of the original Superoptimizer. In particular, this project still performs poorly in terms of time to find a superoptimized solution. Even a 7 sequence of instructions took a “few weeks” to generate, according to the authors.

Another work that builds upon the Superoptimizer is Denali [^3]. Denali changes the way the search space of programs is discovered. Denali only considers a search space of semantically equivalent programs to the original program. In this sense, the search space is smaller than that considered by the original Superoptimizer, because all programs that differ semantically from the original program are omitted. Denali is able to consider this search space by applying rewrite rules to the original program to generate a large search space. These rewrite rules include rules like `2 * x = x + x`. Rewrite rules, which can be either system axioms or user given axioms, are incrementally applied to the program, to create new programs. These new programs are incrementally stored in a data structure, the E Graph, which is able to represent every possible rewrite in a polynomial amount of space. Finally, Denali is able to query the E Graph for programs satisfying certain conditions, which represent optimal programs

Denali operates faster than the original Superoptimizer. Denali is able to generate sequences of instructions ranging up to 31 instructions in length, compared to around 13 for Superoptimizer. Denali also only generates semantically equivalent programs, as compared to Superoptimizer, which might require the user to prove that the generated program is equivalent to the original program. However, the programs that Denali generates are not necessarily minimally sized. In particular, if the minimal size program is not generatable via the rewrite rules that Denali uses, then it cannot be an output of Denali. As a result, there is a tradeoff between speed of the system, versus the ability to completely search the program space to find an optimal solution. Depending on the application, such a tradeoff might be acceptable, particularly when speed of compilation matters more, or when the number of instructions to optimize is larger in size. 

Finally, to improve upon the speed of the Superoptimizer, the STOKE project uses a different search method [^4]. In this project, stochastic search is employed to first search the program space for enumerated programs, and then an optimized program is chosen from these subspaces. This approach is able to scale up to larger programs, using a shorter amount of time to compile, compared to the original Superoptimizer. The authors of STOKE even find that on certain benchmarks, optimized output programs can surpass -O3 and hand optimized programs. Once again, there is a tradeoff where the most minimal program might not be found, as compared to Superoptimizer. However, this tradeoff may not be that important, because STOKE’s ability to beat hand optimized programs, particularly on larger sized programs, allows the ability to bring superoptimization to a larger space of applications, which can benefit more programmers across the spectrum of programming. 

## Conclusion

Massalin’s work is seminal, because it is the first to consider truly optimal programs, rather than incremental improvements that constitute most compiler optimizations. Being able to find optimal real world programs such as the signum function was a major success of Massalin’s work. However, Superoptimizer has several downsides. These downsides are dealt with in other works that build upon ideas found in Superoptimizer. All together, the Superoptimizer provided an intriguing, yet practical approach, to solve an important problem. Today, the techniques pioneered in this paper continue to be used and improved upon to provide new solutions. 

[^1] Henry Massalin. 1987. Superoptimizer – A Look at the Smallest Program. In ACM SIGARCH Computer Architecture News, October 1987. ACM Inc., New York, NY, 122-126. https://doi.org/10.1145/36177.36194

[^2] Torbjorn Granlund, and Richard Kenner. 1992. Eliminating Branches using a Superoptimizer and the GNU C Compiler. In Proceedings of the ACM SIGPLAN 1992 conference on Programming language design and implementation (PLDI ‘92), July 1992. ACM Inc., New York, NY, 341-352. https://dl.acm.org/doi/10.1145/143103.143146

[^3] Rajeev Joshi, Greg Nelson, and Keith H Randall. 2002. Denali: a goal-directed superoptimizer. In Proceedings of the ACM SIGPLAN 2002 conference on Programming language design and implementation (PLDI ‘02), June 2002. ACM Inc., New York, NY, 304-314. https://dl.acm.org/doi/10.1145/512529.512566

[^4] Eric Schkufza, Rahul Sharma, and Alex Aiken. 2013. Stochastic superoptimization. In Proceedings of the eighteenth international conference on Architectural support for programming languages and operating systems (ASPLOS ‘13), March 2013. ACM Inc., New York, NY, 305-316. https://dl.acm.org/doi/10.1145/2499368.2451150