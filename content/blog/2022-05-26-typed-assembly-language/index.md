+++
title = "Type Preserving Compilation to Typed Assembly Language (TAL)"
[extra]
latex = true
[[extra.authors]]
name = "Andrey Yao"
link = "https://github.com/andreyyao"
+++

### Introduction

When it comes to program verification, there are two different approaches that offer correctness guarantee of the compiled assembly. The more obvious way is to verify that the source program and the compilers are both correct. Familiar examples of this include [CompCert](https://compcert.org/) and [CertiCoq](https://certicoq.org/), which are verified compilers for C and Gallina, respectively, as well as [Verifiable C](https://softwarefoundations.cis.upenn.edu/vc-current/index.html), which can be used to check that C source code meets certain specifications.

This project follows the second philosophy which adds extra information to the compiled assembly code, which can then be used to partially verfify the correctness of the assembly. More specifically, this project closely adheres to the semantics and steps defined in Greg Morrisett et al.'s [From System F to Typed Assembly Language](https://dl.acm.org/doi/10.1145/319301.319345). We will be compiling from a strict subset of Standard ML (SML) to a typed assembly language, possibly [Typed Machine Language](https://www.cs.princeton.edu/~appel/papers/tml.pdf) but more likely a custom version of RISC-V like assembly, similar to the one sketched in Morrisett et al.'s paper. In each stage of the compilation including the AST, the IR's, and the final assembly stage, we preserve the type information of the terms in that stage.

Since we need to have type information at each stage, there will be translations of types in addition to translations of expressions. Morrisett et al. proved in their paper the soundness of the typed translations across the entire compilation process. This project is more focused on the implementation, so we will not provide any proofs. However, the intermediate representations we use are closely related to the ones in his paper, so his proof should partially apply here as well.


### Goal and Implementation

The project can be found on [GitHub](https://github.com/andreyyao/tal-riscv). Below is the road map for the project, with check boxes indicating which steps are done and which are not.

- [x] Standard ML
	- [x] language support [specification](https://github.com/andreyyao/tal-riscv/blob/main/specs/sml-support.pdf)
	- [x] AST
		- [x] lexing
		- [x] parsing
		- [x] type checking
	- [ ] IR
		- [x] cps (continuation passing style) translation
		- [ ] closure conversion
		- [ ] abstract assembly
	- [ ] Polymorphic types
- [ ] typed assembly
	- [ ] syntax specification
	- [ ] type checking
	
	
The entire project is written in Rust.

- For the lexing stage, I used [logos](https://docs.rs/logos/latest/logos/), a lexer generator. The tokens are represented as the enum type `Token` in `lex.rs`. The tokens include all the SML keywords, integer and boolean literals, special characters like `(`, `[`, `:`, etc. The tokens also include binary operators `+`, `-`, `*`, as well as built-in functions `not`, `andalso`, `orelse`. Finally, it also supports built-in types `int`, `bool`, `unit` as well as polymorphic type variables.

- For parsing, I wrote a grammar for the subset of SML to work with the parser generator [lalrpop](https://github.com/lalrpop/lalrpop). Since SML is completely specified, I was able to figure out the precedence hierarchy of expressions with not much difficulty. This took a considerable amount of time. As usual, the program is represented as an AST, with a mixture of Rust structs and enums. Since the expression types need to be recursive, some of the struct fields needed to be boxed, like `Box<Expr>`, for example. Finally, each expression variant has a field `typ`, which is initialized to be `Typ::Unknown` in the parser and will be filled in by the type checker as the type of the expression.

- AST type checking is done in the traditional way with a typing context, mapping identifiers to their types. Of course, the context needs to accomodate different "levels" in the AST. Since type checking might be needed for multiple stages in the compilation, I abstracted away a generic context structure.

- The struct `Context<T: Clone>` lives in `util.rs`. It has functions `bind`, `get`, `enter`, and `exeuent`. Think of it as having "levels", corresponding to the parent-child relationship in any tree structure. `bind` binds an identifier to some `T` in the current level. `get` gets the value bound to some identifier. `enter` "saves" the current level and goes down one, and `exeuent` discards the current level and restores the previous level. Each level can see the mappings from all previous levels. For example, to check `e1 + e2` one would wrap the type checking of each of `e1` and `e2` between an `enter` and `exeuent`, and see if both expressions type-checked to `int`.


- CPS translation is fairly standard, and the implementation follows pretty closely to the translation by Morrisett et al. `Expr` is subdivided into two categories: `Expr` and `Value`. `Value` are like expressions but they cannot be further beta-reduced. The types are almost the same, except `Arrow` type is replaced by `Cont`, which doesn't have return types: CPS replaces all returns with a call to a continuation, so every function "returns" void.


These are the features implemented so far. We expect to fill in the missing pieces in the roadmap above.


### Evaluation

Unfortunately we wasn't able to get to assembly generation, so we can't evaluate the implementation fully. However there (fairly comprehensive) integration tests for the first few steps in the compiler, which can be found in the `test` directory.

I also wrote some sample SML programs in the directory `benches`, which can be later be used to test the whole compiler.

To run the tests yourself, simply go to the `compiler` directory and run `cargo t`, provided you have Rust setup.



### Difficulties

There are many unexpected hiccups and difficulties, which was part of the reason why this project is still incomplete so far.

First, choosing the source language was difficult. I went back and forth between typed Racket and SML. The former as a descendant of LISP and Scheme is syntactically closer to System F, and it has explicit type abstraction and type application, which makes it nice. However, its type system, especially the numeric tower and stuff, is a bit too convoluted. SML is more familiar for me since I have programmed in OCaml a lot, but it depends on type inference (unification). I eventually went with SML since its type system makes sense, and to deal with unification I required explicit type annotation in the syntax. However this still leaves out implicit type abstraction and application, which is why I had to disable the polymorphic function feature for the time being.

Another difficulty came from Rust itself. I'm basically new to the language, so getting the borrow checker and lifetime checker to work was a bit tricky. 

Lastly, compilating functional language was fairly new to me. Thus I had to read some sources online to understand how CPS works.


### Self Assessment

Overall, I will rate that the project at its current stage was not successful. This is not to say that it will never be successful, but since it's incomplete you can't really say much about it. Part of the reason was it wasn't finished was that I spent way too much time making the "perfect" plan at the start. In hindsight, I should have settled on something quickly and started churning out Rusty code. There is also unfortunately only 24 hours in the day, which doesn't help.

I wish to keep working on this project over the summer in my free time. If it gets finished, I will update this experience report.
