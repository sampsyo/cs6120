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

## Introduction
The type system of a statically-typed language allows compilers to reject illegal programs during the type-checking stage of compilation. In a sense, the typing information attached to variables, functions, etc. is a refinement on the set of valid program states, which can be approximated without ever running the program. Although programming language types are often used to locate certain errors during compile time, it's not their only use.

"Type-Based Alias Analysis" by Diwan, McKinley, and Moss examines how the same principal can be applied to alias analysis, a type of conservative analysis that determines whether two given pointer variables might interfere with each other, i.e. pointing to the same memory address. Although there had been prior research on alias analysis, Diwan et al.'s type-based alias analysis (TBAA) has the following advantages:
* It is flow-insensitive and runs on linear time, as opposed to many other alias analyses which are expensive to compute.
* It performs almost equally well under an open world assumption as it does in closed world. It's compatible with the principle of modular programming.

They also presented evaluations of TBAA. They performed static and dynamic performance analyses on the effectiveness of TBAA when used for redundant load elimination (RLE). Perhaps most notably, they adopted the strategy of limit analysis by comparing empirical speedups with the maximum possible speedups.

In this blog post, we will first study the specific ideas of TBAA with examples in Java-like syntax, as opposed to Modula-3, the language used in the paper. Then we will review the performance analyses and discuss potential factors behind the empirical results. We will then digress a little bit and talk about extensions to TBAA for more complicated programming language features. Finally we will briefly touch upon the general philosophy of empirical evaluations.

## Type Preliminaries
** Readers can skip this section if they are already familiar with type systems or wish to focus on TBAA **

Given two types $\tau_1$ and $\tau_2$, we say that $\tau_1$ is a subtype of $\tau_2$ if whenever a value of $\tau_2$ is expected, it is legal to supply a value of $\tau_1$ in its place. If we view types as sets and all possible values of a type as its elements, the subtyping relation can be considered roughly the subset relation. Familiar examples from Java include `class Person extends Object` `class LinkedList<T> implements Iterable<T>`, etc. There's also the numeric tower from Typed Racket:
<p align="center">
<img src="numeric_tower.png" alt="alt_text" title="Racket number types" style="zoom:30%;" />
</p>
**St-Amour, Vincent et al. “Typing the Numeric Tower.” PADL (2012).**
TODO add this citation to the end

The subtyping relation is reflexive and transitive. It is an example of a "preorder". We will denote $\tau_1\leq \tau_2$ if the former is a subtype of the latter. There are various ways to construct new subtyping relations given existing ones. For example, if we know that $\tau_1\leq \tau_2$ and $\sigma_1\leq \sigma_2$, it could be reasonable to conclude that the arrow(function) types have $\tau_2\to\sigma_1\leq \tau_1\to\sigma_2$. In this case we say the function type is *covariant* in its return type and *contravariant* in its argument type.

There are other typing rules for constructs like tuples, records, generics, etc., but we will not list all of them here. However, the select examples above already give us a glimpse into the richness of information encoded by types. In general, in a statically-typed type-safe language, stricter typing rules allows more fine-grained TBAA, which we will see shortly.

## Type-Based Alias Analysis
TODO introduce access paths

### Type Declarations Only

### With Field Access

### Extended With Assignments

