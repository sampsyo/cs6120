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

In this blog post, we will first study the specific ideas of TBAA with examples in C#-like syntax, as opposed to Modula-3, the language used in the paper. Then we will review the performance analyses and discuss potential factors behind the empirical results. We will then digress a little bit and talk about extensions to TBAA for more complicated programming language features. Finally we will briefly touch upon the general philosophy of empirical evaluations.

## Type Preliminaries
** Readers can skip this section if they are already familiar with type systems or wish to focus on TBAA **

Given two types $\tau_1$ and $\tau_2$, we say that $\tau_1$ is a subtype of $\tau_2$ if whenever a value of $\tau_2$ is expected, it is legal to supply a value of $\tau_1$ in its place. If we view types as sets and all possible values of a type as its elements, the subtyping relation can be considered roughly the subset relation. Familiar examples from Java include `class Person extends Object` and `class LinkedList<T> implements Iterable<T>`, etc. There's also the numeric tower from Typed Racket:
<p align="center">
<img src="numeric_tower.png" alt="alt_text" title="Racket number types" style="zoom:30%;" />
</p>
**St-Amour, Vincent et al. “Typing the Numeric Tower.” PADL (2012).**
TODO add this citation to the end

The subtyping relation is reflexive and transitive. It is an example of a "preorder". We will denote $\tau_1\leq \tau_2$ if the former is a subtype of the latter. There are various ways to construct new subtyping relations given existing ones. For example, if we know that $\tau_1\leq \tau_2$ and $\sigma_1\leq \sigma_2$, it could be reasonable to conclude that the arrow(function) types have $\tau_2\to\sigma_1\leq \tau_1\to\sigma_2$. In this case we say the function type is *covariant* in its return type and *contravariant* in its argument type.

There are other typing rules for constructs like tuples, records, generics, etc., but we will not list all of them here. However, the select examples above already give us a glimpse into the richness of information encoded by types. In general, in a statically-typed type-safe language, stricter typing rules allows more fine-grained TBAA, which we will see shortly.

## Type-Based Alias Analysis
TBAA operates on the program AST instead of the IR. Thus, it has access to higher level information than some other program analyses. Let's assume the language we're working with has the following kinds of memory references:

1. `a.x`  Class field access
2. `a[n]` Array indexing
3. `*a` Pointer indirection

An access path is defined to be any combination of one or more of these memory references. For instance, `(*(a.b).c[3])[2][*d[*e.f]]` is a pathological example of an access path. Basically, access paths are succinct representations of chains of memory references in the AST. We will also define typeof ($\mathcal{P}$) to be the type of the path $\mathcal{P}$.

### Type Declarations Only

To predict whether two paths $\mathcal{P}_1, \mathcal{P}_2$ might alias, an obvious heuristic is to say this is when the (typeof $\mathcal{P}$) has nonempty intersection with the subtypes of (typeof $\mathcal{P}$). Of course, if the two types are disjoint, then if any expression involving $\mathcal{P}_1$ type checks, the same expression with $\mathcal{P}_2$ substituted in place cannot type check, and so the two paths cannot possibly alias.

### With Field Access

### Extended With Assignments

