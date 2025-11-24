+++
title = "When Prototypes Learned to Run Fast: SELF and the Birth of Adaptive Optimization"
[extra]
latex = true
bio = """
Jake Hyun, Tobi Weinberg, and Adnan Armouti are CS PhD students at Cornell Tech.
"""
[[extra.authors]]
name = "Jake Hyun"
[[extra.authors]]
name = "Tobi Weinberg"
[[extra.authors]]
name = "Adnan Armouti"
+++

# When Prototypes Learned to Run Fast: SELF and the Birth of Adaptive Optimization

*By Jake Hyun, Tobi Weinberg, and Adnan Armouti*

## Introduction: The Core Contribution of SELF

The SELF paper is often misunderstood as focusing purely on language design (prototypes, dynamic typing). Its true contribution lies in being one of the earliest successful attempts to take a **maximally flexible language model and make it fast** using **adaptive optimization**.

Many concepts central to modern Just-In-Time (JIT) compilers—including **speculation**, **inlining**, and **tracking object shapes**—originated here. In 1989, SELF challenged the prevailing wisdom that dynamic languages would inevitably be slow, changing the trajectory of dynamic language performance.

## A Quick Primer on the SELF Language

SELF is characterized by its extreme dynamism and uniformity:
* **Prototypes Only:** No classes are used.
* **Message-Based:** Every field access, and even control structures (like `ifTrue:False:`), is implemented as a **message send**.
* **Runtime Changes:** Inheritance and object structure can change at runtime.

This elegance creates a challenge: a naive implementation would be overwhelmed by the cost of **dynamic lookups**.

## How the System Achieved Performance

The SELF implementation introduced several foundational ideas to manage the late-bound nature of the language.

### 1. Maps (Hidden Classes)

To reclaim efficiency, the system creates **Maps**. These are shared, internal descriptors of an object's memory layout and slot metadata.
* They are essentially the same concept as **hidden classes** in modern JavaScript engines (e.g., V8).
* The implementation silently introduces class-like structure without exposing it to the programmer.

### 2. Customized Dynamic Compilation

Instead of compiling one generic version of a method, SELF compiles **many versions**, specialized for each receiver "shape" (map).
* This specialization allows the compiler to treat the `self` receiver's type as **known**, enabling aggressive **inlining**.

### 3. Message Splitting and Type Prediction

* **Problem:** When control flow merges, type information is lost.
* **Solution:** The compiler handles this by **"splitting"** the following message, creating multiple execution paths.
* The compiler uses **type prediction** to identify common paths (e.g., the result of a comparison is usually `true` or `false`) and generates **fast paths** with guard checks.

### 4. Primitive Inlining

Operations like arithmetic and slot access, which appear as slow message sends in the source code, are recognized and aggressively **inlined** to just a few machine instructions.

### 5. Staying Interactive

Remarkably, SELF maintained a **live programming environment**. It supported:
* **Selective invalidation** when a method was changed.
* **Reconstructing call stacks** for the debugger, even for heavily inlined code.

## Class Discussion and Key Themes

### Flexibility vs. Performance Trade-off
We noted that SELF's extreme flexibility comes at a significant cost, forcing the compiler to work very hard. The discussion questioned if dynamic typing still offers a superior speed advantage for prototyping, given the capabilities of modern static languages and tools (e.g., **TypeScript, Rust, Scala**).

### Missing Costs in Evaluation
The evaluation's focus on small benchmarks was a weakness. The paper largely ignored the critical costs of **compile-time overhead**, **memory use**, and **code-size growth** (due to message splitting), which were especially relevant on 1980s hardware.

### Maps as Implicit Structure
While maps keep the language clean, their implicit nature **hides performance behavior** from the programmer. This means a developer cannot explicitly "lock in" an object's shape to assist the optimizer.

### Legacy and Surviving Ideas
The influence of SELF on modern runtimes is profound:
* **Maps $\rightarrow$ Hidden Classes** (e.g., in V8).
* **Specialized compilation** $\rightarrow$ **Tiered JITs**.
* **Inline Caches** $\rightarrow$ Universal optimization technique.
* **Type prediction** $\rightarrow$ **Speculative optimization**.

The key difference today is **Deoptimization**: modern systems **jump back** to a baseline form when speculation fails, rather than discarding the code entirely.

### Metrics and Relevance
The **MiMS (millions of messages per second)** metric did not impress anyone. It is too tied to the message-passing model. Modern evaluation prioritizes **throughput, latency, warm-up time, and memory footprint**.

### Designing a Modern SELF-like Language
To retain the spirit of SELF while being more practical, the class proposed:
* Keep the **prototype flexibility** and **live-programming feel**.
* Add **optional type hints** and **clearer module boundaries**.
* Allow programmers to explicitly **freeze object shapes** to aid the optimizer.

## Conclusion

The SELF paper's lasting legacy is proving that high-performance is achievable in a highly dynamic model through aggressive, speculative implementation techniques. Its ideas are now **foundational** to how dynamic language VMs are built. The core lesson remains: **performance does not have to dictate language design** if the runtime is smart enough.
