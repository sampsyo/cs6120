+++
title = "Type-based Alias Analysis"
[extra]
latex = true
[[extra.authors]]
name = "Andrew Butt"
link = "TODO"
[[extra.authors]]
name = "Andrey Yao"
link = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
+++

### Introduction
The type system of a statically-typed language allows compilers to reject illegal programs during the type-checking stage of compilation. In a sense, the typing information attached to variables, functions, etc. is a refinement on the set of valid program states, which can be approximated without ever running the program. Although programming language types are often used to locate certain errors during compile time, it's not their only use.

"Type-Based Alias Analysis" by Diwan, McKinley, and Moss examines how the same principal can be applied to alias analysis, a type of conservative analysis that determines whether two given pointer variables might interfere with each other, i.e. pointing to the same memory address. Although there had been prior research on alias analysis, Diwan et al.'s type-based alias analysis (TBAA) has the following advantages:
* It is flow-insensitive and runs on linear time, as opposed to many other alias analyses which are expensive to compute.
* It performs almost equally well under an open world assumption as it does in closed world. It's compatible with the principle of modular programming.

They also presented evaluations of TBAA. They performed static and dynamic performance analyses on the effectiveness of TBAA when used for redundant load elimination (RLE). Perhaps most notably, they adopted the strategy of limit analysis by comparing empirical speedups with the maximum possible speedups.

In this blog post, we will first study the specific ideas of TBAA with examples in Java-like syntax, as opposed to Modula-3, the language used in the paper. Then we will review the performance analyses and discuss potential factors behind the empirical results. We will then digress a little bit and talk about extensions to TBAA for more complicated programming language features. Finally we will briefly touch upon the general philosophy of empirical evaluations.

### Type System Preliminaries
