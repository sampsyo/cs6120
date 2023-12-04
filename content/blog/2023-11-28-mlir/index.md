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

The designers of MLIR claim that their primary motivations for creating this new tool were to solve the problems of software fragmentation and heterogeneous hardware targets. By software fragmentation, the authors mean that compiler engineers working on modern, high-level languages such as Swift, Rust, and Julia have begun creating their own custom, high-level IRs in front of LLVM. This allows compiler engineers to more easily implement source-level optimizations that are significantly more difficult to implement at a lower-level IR. This is because lower-level IRs such as LLVM do not preserve the higher-level semantics that are necessary to more easily implement source-level optimizations. According to the authors, this approach requires excessive engineering resources to build compiler infrastructure that does not generalize to other languages - this is where MLIR comes in. MLIR aims to provide compiler engineers with the freedom to design high-level IRs that allow for source-level optimizations while being able to progressively lower to the typical lower-level IRs such as LLVM, all the while using the same compiler infrastructure.

<img width="699" alt="Screenshot 2023-12-01 at 5 23 21 PM" src="https://github.com/20ashah/cs6120/assets/33373825/2f2e2a84-4e57-4446-aa0d-31a5a9d1a495">

The authors add:

>At the same time, the LLVM community frequently struggled with questions about how to best represent parallel constructs, how to share implementation of common front-end lowering infrastructure (e.g. for C calling conventions, or cross-language features like OpenMP) with no satisfactory solutions being available.

Essentially, the authors were already wrestling with the problem of creating a better compiler infrastructure for machine learning applications when they noticed the fragmentation of high-level IRs. According to the authors, their options were to either develop _N_ improved compiler instances for each source language using a custom, high-level IR or they could develop a new, more general solution. Unsurprisingly, the authors chose the latter.

# TODO: Let's keep each of the below sections brief since according to the blog post guidelines, the summary portion should only be about 25 % of the content and the rest should be our commentary

# MLIR Design Principles 

A few important design principles of MLIR that were highlighted in the discussion were the following:

### Customizability

MLIR has minimal number of built-in features, with things primarily being customizable. Given the goal of creating a generalized IR infrastructure to support high level features of any given domain, this makes sense. MLIR strives to only have a few abstractions which are flexible enough to be reused and express everything that we need. By doing this, we can express a diverse set of abstractions that encompass everything useful in a given domain language, all without hard-coding any of them. 

### Nested regions

Another interesting principle that led to discussion was the shift towards a nested IR over the traditional flat IR implementation. The nested region described in the paper refers to the idea that instead of simply having a sequence of instructions in a flat CFG, we can take a nested approach by having sub-graphs attached to any instruction, allowing for the ability to easily express high level control flow.

### Progressive Lowering

In our discussion, a lot of people brought up some interesting points regarding the lowering approach in MLIR and its advantages over a traditional sequential lowering implementation. It was a consensus that the key here is the fact that MLIR maintains high level semantics. By having lowering take place at multiple abstraction levels allows us to unlock optimizations that would not have been possible with a fixed sequence of passes.

# MLIR Infrastructure

A common question after discussing the goals of MLIR is walking through how the actual implementation of MLIR actually accomplishes these goals. 

First, let's take a look at the difference between the overal structure of LLVM and MLIR and how the subtle changes allow for more flexibility and customization.

### LLVM IR vs MLIR Structure

<img width="290" alt="Screenshot 2023-12-01 at 6 18 10 PM" src="https://github.com/20ashah/cs6120/assets/33373825/65c87ead-7630-4748-bc3d-570cdc8ac1c1"> 

<img width="368" alt="Screenshot 2023-12-01 at 6 18 35 PM" src="https://github.com/20ashah/cs6120/assets/33373825/a4f9bc37-3ad7-46de-98d6-727e752b8ec0">

These structures are very similar in that both structure programs into modules, functions, and blocks, but the way they are implemented in MLIR as "Ops" is an important distinction that contributes to MLIR's extensibility.

### Ops
Ops serve as fundamental computation units within the MLIR. They encapsulate specific functionalities or transformations, offering a high level of abstraction. Ops provide a flexible way to define and customize operations, enabling the representation of diverse domain-specific functionalities and enhancing MLIR's adaptability and expressiveness compared to LLVM's fixed set of instructions. In MLIR, everything is defined as an Op and they can exist at any level of the IR at any time, an advantage that is exploited during progressive lowering.

### Dialects
TODO

# Applications of MLIR

### TensorFlow
TODO


# Discussion

[Discussion thread](https://github.com/sampsyo/cs6120/discussions/419)

There were several discussion topics that came up in class that we will explore further here.

### Role of MLIR in Hardware Heterogeneity

MLIR doesn't directly solve the challenge of heterogeneous hardware, but it paves the way for a potential solution. By providing a uniform intermediate representation, MLIR serves as a bridge between diverse hardware targets and languages. While it doesn't inherently resolve the intricacies of varying hardware architectures, MLIR's modular and extensible nature allows for the creation of custom dialects and transformations. These dialects can encapsulate hardware-specific optimizations, enabling developers to express and apply optimizations relevant to different hardware targets within a unified framework. This approach doesn't eliminate the complexity of heterogeneous hardware but provides a platform where solutions tailored to specific hardware can be developed and integrated more seamlessly. This ties in with several discussion posts about the idea along with its connections to the title regarding the End of Moore's Law. As the industry grapples with the challenge of increasing procesor speed, hardware accelerators emerge as a solution. By enabling the creation of custom dialects and optimizations, MLIR allows developers to harness the full potential of these accelerators while working within the constraints posed by the plateauing of traditional CPU performance growth. Overall, the dicussion on this topic concluded with saying how MLIR opens pathways for tailored solutions across diverse hardware, while stressing that it doesn't resolve the complexities of heterogeneous hardware architectures.


### How good of a solution is MLIR?

MLIR presents a promising avenue for compiler development, offering a versatile framework for expressing diverse transformations and optimizations across different hardware targets. Its modularity and extensibility contribute to its appeal, allowing developers the freedom to craft custom solutions tailored to specific needs. However, amidst its potential, MLIR also comes with limitations. While it seems like a cure for all compiler challenges, it's far from a definitive solution. The abundance of developer freedom within MLIR leads to a lack of standardized best practices, posing a challenge for newcomers navigating its intricacies. To exploit the full potential of MLIR, future research is vital to strike a balance between the expressiveness MLIR offers and the need for optimized performance. Finding this equilibrium will be key to harnessing the full capabilities of MLIR and defining its role in advancing compiler development.

### Limitations

A point of discussion that was brought up in the threads that continued live had to do with certain limitations of MLIR, introducing a note of pessimism and critiques admist the recognition of MLIR's strengths. These limitations revoled around the idea that often times there is a trade off between expressivenesas and performance. In certain instances, while MLIR offers a powerful framework for expressing complex transformations, it doesn't always directly reduce overall complexity. Instead, it might shift the intricacies to another layer or domain within compilation. The trade-offs between expressiveness and performance often constitute a crucial aspect in compiler design and optimization. While enhanced expressiveness within a compiler framework like MLIR allows developers to articulate intricate transformations and optimizations tailored to specific requirements, it might inadvertently introduce complexities that impact performance. This trade-off involves finding a delicate balance: the more expressive the framework, the greater the potential for sophisticated optimizations, but this might come at the cost of increased compilation time or overhead in the generated code.
