+++
title = "Simple and Efficient Construction of Static Single Assignment Form"
[[extra.authors]]
name = "Dev Patel"
[[extra.authors]]
name = "Neel Patel"
+++
# Background

Single static assignment (SSA) form [[1]](#1) and the first efficient conversion algorithm [[2]](#2) emerged in the 1980's. SSA is an intermediate representation (IR) which enforces that each variable is assigned exactly once. It provides a convenient form and way of thinking about programs which makes some optimizations more efficient.

<img src="image-27.png" alt="" width="60%">

The first published algorithm for SSA construction [[6]](#6), written by Cytron et al., proceeds in two steps. The first places phi functions throughout the program, indicating ambiguities in assignments due to control flow. The second renames variables to ensure SSA’s single assignment property is satisfied. Importantly, Cytron’s algorithm relies on calculation of the dominance frontier: “the set of all CFG nodes Y such that X dominates a predecessor of Y but does. not strictly dominate Y”, for each basic block X in the control flow graph (CFG) representation of the program.

<img src="image-28.png" width="60%" alt="">

# Contributions of Braun et al.’s algorithm [[5]](#5)

In “Simple and Efficient Construction of Static Single Assignment Form” Braun et al. present an alternative algorithm for SSA construction. The main benefit to their algorithm is that it goes straight from the abstract syntax tree (AST) representation of a program to SSA form, and eschews the calculation of auxiliary data structures, like the dominance frontier.

The algorithm calculates phi nodes lazily using recursion. The main steps are (1) local value numbering to lookup values defined in the same basic block and (2) global value numbering to recursively lookup values defined in the predecessors of a basic block.

<div>
  <img src="image-29.png" alt="alt text">
</div>
<div>
  <img src="image-30.png" alt="alt text">
</div>

During global value numbering trivial phi functions that just reference themselves and one other value are removed. The algorithm also enables local, on-the fly optimizations such as constant folding, copy propagation, arithmetic simplification, and common subexpression elimination. If arbitrary control flow is possible (e.g., goto statements), strongly connected components of redundant phi functions (groups of phis that only reference each other and one other incoming definition from outside the group) are also removed.
# Merits and shortcomings
### Merits
There are many positive claims about this algorithm that were made both in the paper and during the discussion section. By eliminating the need to construct dominance data structures, compile times are faster for specific scenarios. The combining of two transformations into one removes any detours along the pipeline. This direct translation also reduces the memory overhead- these data structures computed during the intermediate steps no longer need to be stored in memory. This algorithm could be particularly useful for Just-In-Time compilation contexts. These are situations where the compiler is less likely to heavily rely on dominance information for the optimizations, but it will still utilize SSA form. This is an example of a specific case where the direct translation has clear benefits- faster SSA conversion, no detours and less memory overhead.

### Shortcomings
Although these merits seem enticing, there are several shortcomings. The algorithm results in much more complexity than what the paper emphasizes. This is especially apparent in the context of compiler programmers working with this direct translation approach but there is also complexity within the actual algorithm regarding mechanisms such as sealing blocks. The main concern, however, is about the maintainability and potentially leading to many bugs within the actual optimization passes. Additionally, in many cases our future passes will rely on dominance data structures. In these cases, the intermediate steps of SSA conversion are not truly detours as the side effects bring value to a future optimization pass. It should be noted that these data structures may not be fully accurate in the case that the SSA conversion results in a global optimization that changes the CFG, but it is still presumably more efficient to adjust these dominance structures for accuracy rather than creating or recreating them later. This application seems like it will only be valuable when applied to certain use cases. It also may create a lot of implementation overhead. For example, we may need to implement lots of boilerplate for the syntax of different languages when trying to complete this direct translation which can make the overall optimization go slower. This can particularly be the case for compilers with frontends that support multiple languages. The largest source of skepticism lied within the empirical results presented in the paper. The paper made loose claims about “optimized” and “unoptimized” versions of their algorithm and Cytron et al. 's algorithm. This led to doubts about whether the performance will actually improve in a real-world compilation or if this is a project that emphasizes only theoretical benefits.
# Historical Context
Prof. Sampson mentioned that SSA form [[1]](#1) and the first efficient conversion algorithm [[2]](#2) emerged in the 1980's, whereas the simple and efficient algorithm discussed in class was published in 2013. One question discussed in class was why Cytron et al.'s implementation has been the de facto SSA conversion scheme, used in the LLVM IR [[3]](#3) and other languages’ compiler toolchains [[4]](#4). Its ~25 year head start is one likely reason.
It is yet to be seen whether the efficient algorithm implemented by Braun et al. [[5]](#5) will find applications in any compiler toolchains.

# References
<a id="1">[1]</a>
https://compilers.cs.uni-saarland.de/ssasem/talks/Kenneth.Zadeck.pdf

<a id="2">[2]</a>
http://www.cs.utexas.edu/~pingali/CS380C/2010/papers/ssaCytron.pdf

<a id="3">[3]</a>
https://llvm.org/docs/LangRef.html

<a id="4">[4]</a>
https://internals.rust-lang.org/t/why-does-rustc-implement-ssa/21177

<a id="5">[5]</a>
https://c9x.me/compile/bib/braun13cc.pdf

<a id="6">[6]</a>
https://dl.acm.org/doi/pdf/10.1145/75277.75280
