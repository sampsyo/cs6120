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

# ***Recap begins here***

# Summary
[Slides from discussion](https://docs.google.com/presentation/d/1dHY8Xrk-VhUodql-06egCotdDWsIoiNOgEzlmSc0coM/edit?usp=sharing)

The designers of MLIR claim that their primary motivations for creating this new tool were to solve the problems of software fragmentation and heterogeneous hardware targets. By software fragmentation, the authors mean that compiler engineers working on modern, high-level languages such as Swift, Rust, and Julia have begun creating their own custom, high-level IRs in front of LLVM. This allows compiler engineers to more easily implement source-level optimizations that are significantly more difficult to implement using a lower-level IR. This is because lower-level IRs such as LLVM IR do not preserve the higher-level semantics that are necessary to more easily implement source-level optimizations. According to the authors, this approach requires excessive engineering resources to build compiler infrastructure that does not generalize to other languages - this is where MLIR comes in. MLIR aims to provide compiler engineers with the freedom to design high-level IRs that allow for source-level optimizations while being able to progressively lower to the typical lower-level IRs such as LLVM IR, all the while using the same compiler infrastructure.

<img width="699" alt="Screenshot 2023-12-01 at 5 23 21 PM" src="https://github.com/20ashah/cs6120/assets/33373825/2f2e2a84-4e57-4446-aa0d-31a5a9d1a495">

The authors add:

>At the same time, the LLVM community frequently struggled with questions about how to best represent parallel constructs, how to share implementation of common front-end lowering infrastructure (e.g. for C calling conventions, or cross-language features like OpenMP) with no satisfactory solutions being available.

Essentially, the authors were already wrestling with the problem of figuring out how to create a better compiler infrastructure for machine learning applications when they noticed the fragmentation of high-level IRs. According to the authors, their options were to either develop _N_ improved compiler instances for each source language using a custom, high-level IR or they could develop a new, more general solution. Unsurprisingly, the authors chose the latter.

# MLIR Design Principles 

MLIR was constructed based on a set of core design principles. A few key design principles that were highlighted in the discussion were the following:

### Customizability

To allow compiler engineers to design a wide range of IRs, MLIR needed to be sufficiently customizable. Therefore, MLIR has very little built-in. The idea is that the built-in features are extremely flexible, providing users with a high degree of expressivity in the IRs they build. In theory, customizability promotes code reuse and allows MLIR users to work with "dialects" that maintain the high-level semantics which make source-level optimizations easier to implement. 

### Nested Regions

A key approach the authors take to promoting the flexibility of MLIR is the _nested regions_ feature. The authors state:

>While many existing IRs use a [flat]([url](https://www.cs.cornell.edu/~asampson/blog/flattening.html)), linearized CFG, representing higher level abstractions push introducing _nested regions_ as a first-class concept in the IR. This goes beyond the traditional region formation to lift higher level abstractions (e.g., loop trees), speeding up the compilation process or extracting instruction, or SIMD parallelism. To support heterogeneous compilation, the system has to support the expression of structured control flow, concurrency constructs, closures in source languages, and many other purposes. One specific challenge is to make CFG-based analyses and transformations compose over nested regions.

In short, MLIR's focus on nested regions allows for the representation of higher-level abstractions, such as loop trees, in a structured manner. The challenge lies in ensuring that CFG-based analyses and transformations can effectively compose over these nested regions, especially to support heterogeneous compilation for various purposes like structured control flow, concurrency constructs, and closures in source languages.

### Progressive Lowering

MLIR emphasizes flexibility as one of its core design principles, enabling compiler engineers to create multiple high-level IRs in front of lower-level IRs like LLVM IR. The approach to supporting an arbitrary number of IRs is through a concept called "progressive lowering." In the compilation pipeline, each higher-level IR, or dialect, undergoes a gradual lowering process to the next level, simplifying the task of writing passes. Of note, many existing compilers supported progressive lowering across a _fixed_ number of IRs at the time of MLIR's development. Part of what sets MLIR apart from previous compilers is its ability to support progressive lowering across an arbitrary number of IRs.

# MLIR Infrastructure

How does the implementation of MLIR achieve its design principles? 

First, let's take a look at the difference between the overall structure of LLVM and MLIR and how subtle changes allow for more flexibility and customization.

### LLVM IR vs MLIR Structure

<img width="290" alt="Screenshot 2023-12-01 at 6 18 10 PM" src="https://github.com/20ashah/cs6120/assets/33373825/65c87ead-7630-4748-bc3d-570cdc8ac1c1"> 

<img width="368" alt="Screenshot 2023-12-01 at 6 18 35 PM" src="https://github.com/20ashah/cs6120/assets/33373825/a4f9bc37-3ad7-46de-98d6-727e752b8ec0">

These structures are very similar in that both structure programs into a hierarchy of modules, functions, and blocks, but the way they are implemented in MLIR as "Operations" is an important distinction that contributes to MLIR's extensibility.

### Operations
Operations, or Ops, serve as the fundamental semantic unit within the MLIR. Everything in MLIR is an Op. This includes every structure in MLIR's hierarchy including modules, functions, blocks, and instructions. Ops were specifically designed to allow for user-defined extensions and the structure of an Op is how MLIR's customizability design principle is achieved. 

### Dialects
MLIR's extensibility and support for an arbitrary number of IRs is realized through the use of _Dialects_. Dialects serve as logical groupings for Ops, attributes, and types, analogous to modular libraries in programming languages like C++. Part of MLIR's strength lies in its ability to mix dialects, enabling the coexistence of Ops from different dialects with Ops from any level IR. This facilitates the preservation of higher-level semantics throughout the compilation pipeline until they are no longer required.

# Applications of MLIR

### TensorFlow
TODO - Jiahan

# ***Recap ends here***

# TODO: Revise / add on below

Possible Talking Points - 

1) Despite the title, MLIR is useful for more than just "post-Moore's Law" challenges (e.g. Rust's borrow checking, polyhedral compilation, how MLIR makes it easier for functional languages to compile to LLVM IR)
2) How much does MLIR really help with the machine learning applications it was (partially) designed to help solve?
3) With MLIR available, is there still a place for custom, high-level IRs?
4) What do we think of MLIR using nested IRs which is a divergence from the more recent trend of flat IRs?

# Discussion

[Discussion thread](https://github.com/sampsyo/cs6120/discussions/419)

There were several discussion topics that came up in class that we will explore further here.

### Role of MLIR in Hardware Heterogeneity <- Maybe we should provide a brief description of what the challenges of heterogeneous compilation are and why MLIR doesn't necessarily solve them but perhaps will make it easier to solve this problem

MLIR doesn't directly solve the challenge of heterogeneous hardware, but it paves the way for a potential solution. By providing a uniform intermediate representation, MLIR serves as a bridge between diverse hardware targets and languages. While it doesn't inherently resolve the intricacies of varying hardware architectures, MLIR's modular and extensible nature allows for the creation of custom dialects and transformations. These dialects can encapsulate hardware-specific optimizations, enabling developers to express and apply optimizations relevant to different hardware targets within a unified framework. This approach doesn't eliminate the complexity of heterogeneous hardware but provides a platform where solutions tailored to specific hardware can be developed and integrated more seamlessly. This ties in with several discussion posts about the idea along with its connections to the title regarding the End of Moore's Law. As the industry grapples with the challenge of increasing processor speed, hardware accelerators emerge as a solution. By enabling the creation of custom dialects and optimizations, MLIR allows developers to harness the full potential of these accelerators while working within the constraints posed by the plateauing of traditional CPU performance growth. Overall, the discussion on this topic concluded with saying how MLIR opens pathways for tailored solutions across diverse hardware while stressing that it doesn't resolve the complexities of heterogeneous hardware architectures.


### How good of a solution is MLIR?

MLIR presents a promising avenue for compiler development, offering a versatile framework for expressing diverse transformations and optimizations across different hardware targets. Its modularity and extensibility contribute to its appeal, allowing developers the freedom to craft custom solutions tailored to specific needs. However, amidst its potential, MLIR also comes with limitations. While it seems like a cure for all compiler challenges, it's far from a definitive solution. The abundance of developer freedom within MLIR leads to a lack of standardized best practices, posing a challenge for newcomers navigating its intricacies. To exploit the full potential of MLIR, future research is vital to strike a balance between the expressiveness MLIR offers and the need for optimized performance. Finding this equilibrium will be key to harnessing the full capabilities of MLIR and defining its role in advancing compiler development.

### Limitations

A point of discussion that was brought up in the threads that continued live had to do with certain limitations of MLIR, introducing a note of pessimism and critiques amidst the recognition of MLIR's strengths. These limitations revolved around the idea that oftentimes there is a trade-off between expressiveness and performance. In certain instances, while MLIR offers a powerful framework for expressing complex transformations, it doesn't always directly reduce overall complexity. Instead, it might shift the intricacies to another layer or domain within compilation. The trade-offs between expressiveness and performance often constitute a crucial aspect in compiler design and optimization. While enhanced expressiveness within a compiler framework like MLIR allows developers to articulate intricate transformations and optimizations tailored to specific requirements, it might inadvertently introduce complexities that impact performance. This trade-off involves finding a delicate balance: the more expressive the framework, the greater the potential for sophisticated optimizations, but this might come at the cost of increased compilation time or overhead in the generated code.
