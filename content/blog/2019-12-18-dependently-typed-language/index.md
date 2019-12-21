+++
title = "A Dependently Typed Language"
[extra]
bio = """
  Christopher Roman is a second semester MEng student in computer science. He is interested in compilers and distributed systems. He is also the best [Melee](https://en.wikipedia.org/wiki/Super_Smash_Bros._Melee) player at Cornell. :)
"""
latex = true
[[extra.authors]]
name = "Christopher Roman"
+++

## Overview
For this project, we implement the [Calculus of Constructions](https://www.sciencedirect.com/science/article/pii/0890540188900053) (CoC)
which is a lambda calculus that sits at the top of the [Lambda Cube](https://en.wikipedia.org/wiki/Lambda_cube). This means that
it includes polymorphism, type level operators, and dependent types. With the Simply Typed Lambda Calculus, terms are only
allowed to depend on terms. Polymorphism lets terms depend on types (e.g., generics). Type level operators allow types to
depend on types (e.g., `list` in OCaml). Dependent types allow types to depend on terms. Dependent types are rather
uncommon in most programming languages, so I'd suggest reading [these notes](https://www.cs.cornell.edu/courses/cs4110/2018fa/lectures/lecture31.pdf)
created by our very own [Adrian Sampson](https://www.cs.cornell.edu/~asampson/) for an introduction to how dependent types work.

Ultimately, dependent types allow us to make full use of the Curry-Howard ismorphism; that is, types correspond to
logical statements, and programs correspond to proofs of such statements. For example, consider the polymorphic
identity function $\Lambda A. \lambda x: a. x$, or written `[A: *][x: A]x` in CoC. The type of this program
is written `[A: *][x: a]a`, which represents the logical statement $\forall A. a \implies a$. By the Curry-Howard isomorphism,
the identity function serves as a proof for that statement because it inhabits the corresponding type. Through this,
we can write proofs and be assured that they are free of mistakes (or at least moreso than hand-written proofs).

The Calculus of Constructions provides a quite simple type system that allows us to write proofs through programming.
Our goal is to implement CoC and show the ability to write some proofs.

## Design
Some of the design decisions about how to write CoC programs came from an [existing implementation](https://github.com/lambda-11235/ttyped)
of CoC. That implementation is well written and good for working with CoC, though it uses a *slightly* different syntax.

### The Grammar
[This paper](https://www.cs.cmu.edu/~fp/papers/mfps89.pdf) provides a concise grammar for CoC:
TODO: Insert image instead for grammar
```
M ::= x | (\x: M)N | (M N) | [x: M]N | *
```

### Typing Rules
CoC makes a distinction between *contexts* and *objects*. Contexts are products over `*`, that is, terms
of the form $[x_1 : M_1] [x_2 : M_2] ... [x_n : M_n] *$. These terms are denoted as $\Gamma$ and $\Delta$.

### Additions to CoC

## Implementation

## Evaluation

## Hardest Parts to Get Right
