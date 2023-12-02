+++
title = "MLIR: A Compiler Infrastructure for the End of Moore’s Law"
[[extra.authors]]
name = "John Rubio"
[[extra.authors]]
name = "Jiahan Xie"
[[extra.authors]]
name = "Arjun Shah"
[extra]
latex = true
+++

# Summary
[Slides from discussion](https://docs.google.com/presentation/d/1dHY8Xrk-VhUodql-06egCotdDWsIoiNOgEzlmSc0coM/edit?usp=sharing)

The main motivator behind MLIR is the problem that arises when we want to deal with langugage-specific optimizations during compilation. LLVM does not include any high level semantics into its IR, which allows it to be so great at general optimziations regardless of the domain. However, this utility comes with the downside of being unable to easily deal with domain specific optimizations. A solution (which was adopted by several high langages such as Swift and Rust), included inventing a specialized IR for a given domain that would incldue high level aspects of the language before compiling down to LLVM IR.

<img width="699" alt="Screenshot 2023-12-01 at 5 23 21 PM" src="https://github.com/20ashah/cs6120/assets/33373825/2f2e2a84-4e57-4446-aa0d-31a5a9d1a495">

An issue with this approach is that it requires a lot of redundency when creating a new specialized IR that may make it not worth the engineering effort. MLIR is a more generalized infrastructure to avoid this duplication while still allowing for the inclusion of high level semantics through customization to support domain specific optimizations.

# MLIR Design Principles

A few important design principles of MLIR that were highlighted in the discussion were the following:

### Customizability

MLIR has minimal number of built-in features, with things primarily being customizable. Given the goal of creating a generalized IR infrastructure to support high level features of any given domain, this makes sense. MLIR strives to only have a few abstractions which are flexible enough to be reused and express everything that we need. By doing this, we can express a diverse set of abstractions that encompass everything useful in a given domain language, all without hard-coding any of them. 

### Nested regions

Another interesting principle of the paper that we discussed in depth was the idea that MLIR is moving away from flat CFGs by allowing a nested IR structure.


### Maintaining High Level Semantics

### Progressive Lowering

# MLIR Infrastructure

A common question after discussing the goals of MLIR is walking through how the actual implementation of MLIR actually accomplishes these goals. 

First, let's take a look at the difference between the overal structure of LLVM and MLIR and how the subtle changes allow for more flexibility and customization.

### LLVM IR vs MLIR Structure

<img width="290" alt="Screenshot 2023-12-01 at 6 18 10 PM" src="https://github.com/20ashah/cs6120/assets/33373825/65c87ead-7630-4748-bc3d-570cdc8ac1c1"> 

<img width="368" alt="Screenshot 2023-12-01 at 6 18 35 PM" src="https://github.com/20ashah/cs6120/assets/33373825/a4f9bc37-3ad7-46de-98d6-727e752b8ec0">

These structures are very similar in that both structure programs into modules, functions, and blocks, but the way they are implemented in MLIR as "Ops" is an important distinction that contributes to MLIR's extensibility.

### Ops


# Applications of MLIR

### TensorFlow

### Fortran IR

### ClangIR

# Conclusions / Future Research

Limitations?
