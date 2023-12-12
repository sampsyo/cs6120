+++
title = "Implementing the Polyhedral Model"
[extra]
latex = true
[[extra.authors]]
name = "John Rubio"
[[extra.authors]]
name = "Arjun Shah"
+++

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

### MLIR & TensorFlow

MLIR aids the development of machine learning framework, such as TensorFlow, which typically utilize data flow graphs with dynamic execution semantics. An example of such a framework is TensorFlow.

The TensorFlow ecosystem includes compilers and optimizers functioning across software and hardware levels, often resulting in complex errors when using different hardwares. To address these issues, MLIR is used to bridge the gap between model representation and hardware-specific code, simplifying compiler design and optimize performance. MLIR is flexible, allowing custom types in dialects and facilitating the integration of new compilers or hardware by creating new dialects, including TensorFlow IR, XLA HLO IR, affine dialect, etc.

# Discussion

### MLIR is more than a solution to post-Moore's Law challenges

Despite the title of the paper, MLIR's use cases are far broader than a solution to "post-Moore's Law" challenges such as heterogeneous hardware compilation. Here, we discuss two applications of MLIR which are simply scenarios in which having a flexible, extensible IR comes in handy - nothing about these applications relates to the end of Moore's Law.

#### MLIR for Polyhedral Compilation

Polyhedral compilation is an advanced compilation technique that focuses on optimizing programs with multidimensional arrays and loop nests. It is particularly effective for scientific and numerical computing applications that involve regular, dense computations, such as simulations, image processing, and linear algebra operations.

Since polyhedral compilation uses many high-level abstractions, it has often been useful to define custom, high-level IRs in front of LLVM IR to help bridge the semantic gap. Polyhedral-specific dialects can be created within MLIR to capture the semantics and transformations associated with polyhedral compilation. This allows for a clean and expressive representation of polyhedral concepts which can be progressively lowered and optimized over several passes before reaching a low-level IR such as LLVM IR.

#### MLIR for Compiling Functional Languages

MLIR can be beneficial for compiling functional languages due to its flexibility, extensibility, and support for expressing high-level abstractions. For example, it is not difficult to imagine how MLIR's custom dialects make it easier for compiler engineers to represent high-level, abstract features such as immutability, higher-order functions, and closures. Additionally, functional languages often have unique constructs like algebraic data types, pattern matching, and lazy evaluation. Such abstract, language-specific features are perfect examples of when it is a good idea to define a dialect in MLIR. Another challenge of implementing functional languages is lowering high-level, abstract, functional constructs to a lower level IR such as LLVM IR. LLVM IR is more suitable for imperative source languages since LLVM was originally designed with C and C++ in mind. Thus, MLIR's progressive lowering capability can make the job of lowering to LLVM IR from a functional language much less daunting.

### MLIR's Role in Heterogeneous Compilation 

#### What is Heterogeneous Compilation?
Heterogeneous compilation refers to the process of compiling code that is designed to run on different types of hardware architectures or devices. In a heterogeneous computing environment, various types of processing units, such as CPUs, GPUs, FPGAs, and specialized accelerators, may be present. Heterogeneous compilation aims to generate optimized code for each specific hardware target within the same program or application.

The goal is to take advantage of the strengths and capabilities of different hardware architectures, optimizing the code for parallelism, concurrency, and specific features of each processing unit. This approach allows developers to harness the full potential of diverse hardware components within a single application, enhancing performance and efficiency.

Suppose you have a gradient descent program written in C++ that you want to compile and optimize for both a CPU and a GPU, leveraging the strengths of each architecture. This scenario exemplifies heterogeneous compilation.

In a heterogeneous compilation workflow:

1. **CPU Compilation:**
   - The original C++ code is first compiled into an intermediate representation suitable for a CPU. This might involve optimizing for the architecture-specific features of the CPU.
   - CPU-specific optimizations are applied to improve performance on traditional central processing units.

2. **GPU Compilation:**
   - The same C++ code can be compiled into a different intermediate representation suitable for a GPU.
   - GPU-specific optimizations are applied to exploit parallelism and take advantage of the massively parallel architecture of the GPU.

3. **Heterogeneous Execution:**
   - The compiled CPU and GPU code can be combined into a single executable or run as separate components within the same program.
   - The program can dynamically decide at runtime which portions of the computation are best suited for execution on the CPU or GPU, based on the available hardware resources.

In this example, heterogeneous compilation enables the generation of optimized code for both CPU and GPU architectures, allowing the program to run efficiently on diverse hardware. The ability to express and optimize for different hardware targets is crucial in modern computing environments where applications may need to leverage a variety of processing units for performance gains.

#### Where does MLIR come in?

MLIR doesn't directly solve the challenge of heterogeneous hardware, but it paves the way for a potential solution. By providing a uniform intermediate representation, MLIR serves as a bridge between diverse hardware targets and languages. While it doesn't inherently resolve the intricacies of varying hardware architectures, MLIR's modular and extensible nature allows for the creation of custom dialects and transformations. These dialects can encapsulate hardware-specific optimizations, enabling developers to express and apply optimizations relevant to different hardware targets within a unified framework. This approach doesn't eliminate the complexity of heterogeneous hardware but provides a platform where solutions tailored to specific hardware can be developed and integrated more seamlessly. This ties in with several discussion posts about the idea along with its connections to the title regarding the End of Moore's Law. As the industry grapples with the challenge of increasing processor speed, hardware accelerators emerge as a solution. By enabling the creation of custom dialects and optimizations, MLIR allows developers to harness the full potential of these accelerators while working within the constraints posed by the plateauing of traditional CPU performance growth. Overall, the discussion on this topic concluded with saying how MLIR opens pathways for tailored solutions across diverse hardware while stressing that it doesn't resolve the complexities of heterogeneous hardware architectures.

###  Nested IRs vs. Flat IRs

The addition of a nested IR approach within MLIR is due to the need for greater expressivity and domain-specific optimizations, allowing for the ability to represent complex semantics and control flow that is difficult to do in a traditional flat IR. The nested approach in lowering facilitates gradual transformations across various abstraction levels, enabling tailored optimizations for diverse domains. Importantly, this shift towards nested IRs doesn't mean completely moving away from flat IRs but rather introducing a middle layer that allows us to go from MLIR to a traditional flat IR, leveraging the advantages of both. This hybrid approach balances expressivity with computational efficiency, offering a middle ground that allows for tailored optimizations without disregarding the benefits of flat IRs.

### Does MLIR eliminate the need for custom, high-level IRs?

Despite all of the strengths of MLIR and how it solves the problem of having to introduce high-level IRs for each domain, this doesn't mean that high-level IRs no longer have a place in compiler development. There are plenty of instances where it may make sense to use a custom IR rather than MLIR - one being in the design of specialized hardware. Designing custom hardware may require precise control over low-level details. Having a custom IR tailored to the specific needs of the IR may provide a level of control at the hardware level that is hard to accomplish with MLIR. While MLIR offers a flexible framework for representing different IRs, its focus might not align perfectly with the intricacies and low-level optimizations required in hardware description. Designing a custom IR tailored specifically for hardware description could provide developers with the precise control and optimizations needed to generate efficient hardware designs.

### How good of a solution is MLIR?

MLIR presents a promising avenue for compiler development, offering a versatile framework for expressing diverse transformations and optimizations across different hardware targets. Its modularity and extensibility contribute to its appeal, allowing developers the freedom to craft custom solutions tailored to specific needs. Its nested-IR design increases flexibility and extensibility since it can represent different levels of abstraction more naturally, accommodating a wide range of hardware targets and optimization levels. This flexibility is crucial in the domain of machine learning and heterogeneous computing, where new hardware architectures and optimization techniques are continually emerging. However, amidst its potential, MLIR also comes with limitations. While it seems like a cure for all compiler challenges, it's far from a definitive solution. The abundance of developer freedom within MLIR leads to a lack of standardized best practices, posing a challenge for newcomers navigating its intricacies. It is also noteworthy that while nested IRs provide more expressive power, they also introduce additional complexity in terms of representation and processing. Compilers must manage and navigate these nested structures, which can be more challenging than handling flat IRs. The multi-level nature of MLIR's nested IRs can potentially lead to more efficient compilation and execution, especially for complex, domain-specific tasks. However, the added complexity of navigating nested structures could also impact compilation times or runtime performance, depending on the implementation and use cases. To exploit the full potential of MLIR, future research is vital to strike a balance between the expressiveness MLIR offers and the need for optimized performance. Finding this equilibrium will be key to harnessing the full capabilities of MLIR and defining its role in advancing compiler development.

### Limitations

MLIR is not a silver bullet and it does have its limitations. These limitations revolve around the idea that oftentimes there is a trade-off between expressiveness and performance. In certain instances, while MLIR offers a powerful framework for expressing complex transformations, it doesn't always directly reduce overall complexity. Instead, it might shift the intricacies to another layer or domain within the compilation process. The trade-offs between expressiveness and performance often constitute a crucial aspect in compiler design and optimization. While enhanced expressiveness within a compiler framework like MLIR allows developers to articulate intricate transformations and optimizations tailored to specific requirements, it might inadvertently introduce complexities that impact performance. This trade-off involves finding a delicate balance: the more expressive the framework, the greater the potential for sophisticated optimizations, but this might come at the cost of increased compilation time or overhead in the generated code.

---

_John Rubio is a 2nd year MS student at Cornell University. He is interested in compilers, programming languages, and hardware._

_Arjun Shah is a senior undergraduate at Cornell University. He is interested in working on compilers in industry._

_Jiahan Xie is a 1st year MS student at Cornell University. He is interested in compilers and LLVM._
