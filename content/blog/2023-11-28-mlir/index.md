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

Another interesting principle that led to discussion was the shift towards a nested IR over the traditional flat IR implementation. The nested region described in the paper refers to the idea that instead of simply having a sequence of instructions in a flat CFG, we can take a nested approach by having sub-graphs attached to any instruction. A point was brought up in discussion that it seems strange to do it this way and almost seems like a step back by going against the traditional flat IR. However, when we consider the overall progressive lowering approach happening that multiple abstraction levels, it starts to make perfect sense. In addition, a point was made that this also doesn't mean we are moving away from flat IRs, but rather introducing an IR that is in between to express high level control flow easier.

### Progressive Lowering

In our discussion, a lot of people brought up some interesting points regarding the lowering approach in MLIR and its advantages over a traditional sequential lowering implementation. It was a consensus that the key here is the fact that MLIR maintains high level semantics. By having lowering take place at multiple abstraction levels allows us to unlock optimizations that would not have been possible with a fixed sequence of passes like in LLVM. There were lots of good talking points surrounding this idea during the discussion in that given that we have high level semantics in MLIR, it makes perfect sense to not only gradually lower at these abstraction levels but also mix and match between these abstraction levels before going down to generic LLVM IR. This idea is made possible through how MLIR is actually designed, which the paper then transitions to.

# MLIR Infrastructure

A common question after discussing the goals of MLIR is walking through how the actual implementation of MLIR actually accomplishes these goals. 

First, let's take a look at the difference between the overal structure of LLVM and MLIR and how the subtle changes allow for more flexibility and customization.

### LLVM IR vs MLIR Structure

<img width="290" alt="Screenshot 2023-12-01 at 6 18 10 PM" src="https://github.com/20ashah/cs6120/assets/33373825/65c87ead-7630-4748-bc3d-570cdc8ac1c1"> 

<img width="368" alt="Screenshot 2023-12-01 at 6 18 35 PM" src="https://github.com/20ashah/cs6120/assets/33373825/a4f9bc37-3ad7-46de-98d6-727e752b8ec0">

These structures are very similar in that both structure programs into modules, functions, and blocks, but the way they are implemented in MLIR as "Ops" is an important distinction that contributes to MLIR's extensibility.

### Ops
TODO

### Dialects
TODO

# Applications of MLIR

### TensorFlow
TODO

# Conclusions / Future Research

A lot of our discussion focused on the future of MLIR given that this is a relatively new idea. An interesting point that was brought up in the discussion was that to an outside perspective of someone not in compilers, MLIR seems like a groundbreaking tool that makes all compiler development trivial, when in reality this is not the case. An important part of the paper that we discussed was the mention of how there was little guidance in best practices of using MLIR given that there is so much developer freedom. The consensus was that to exploit the full potential of MLIR is going to require more research and finding a balance between expressiveness and performance.
