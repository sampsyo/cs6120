+++
title = "A Semi-Practical Type-and-Effect System"
[extra]
bio = """
  Kenneth Li is a first-semester M.Eng. student who is interested in functional
  programming. He sings bass in the 
  [Cornell Hangovers](http://www.hangovers.com/).
"""
[[extra.authors]]
name = "Kenneth Li"
+++

# Introduction
Side effects in programming are any observable effects caused by executing some code beyond its return values. For example, the simple `print` functions inherent in most languages cause the side effect of printing to the standard output stream. However, because side effects can be observed from the outside, such as by the program user, then to enforce determinism of programs the order of side-effect-causing instructions must be the same. However, side-effect-free instructions, which we will refer to as "pure", can be rearranged arbitrarily or even precomputed, so long as the code results in the same output. Unfortunately, in most modern programming languages, it is difficult to know what side effects are caused by what code, making it difficult for compilers to reorder instructions. In this report, we explore a way to encode these side effects in a static manner.

In order to perform side-effect-aware static analysis, we use the idea of a _type-and-effect system_ (as explored in [1](https://www.janestreet.com/tech-talks/effective-programming/), [2](http://web.cs.ucla.edu/~todd/research/tldi09.pdf), and [3](https://www.ccs.neu.edu/home/amal/course/7480-s12/effects-notes.pdf)), which extends the notion of types to encode side effect information. For example, the OCaml function `print_int` has type `int -> unit`, but in a type-and-effect system, the signature might look more like `print_int : (int -> unit !output)`. The following section explores one implementation of such a system, available [here](https://github.com/kq-li/type-and-effect-system).

# Language
The language designed for this project operates over four types -- `unit`, `int`, `bool`, and one-argument functions `t1 -> t2` -- and four effects -- `input`, `output`, `read`, and `write`. In addition, the language includes control flow statements such as conditionals and loops. Finally, the language supports higher-order and anonymous functions.

The standard library for the language is composed of the following functions:
- boolean operations `not`, `and`, and `or`
- arithmetic operations `neg`, `add`, `sub`, `mul`, `div`, and `mod`
- comparison operations `lt`, `le`, `gt`, `ge`, `eq`, and `neq`
- side-effecting operations
    - `scan : (unit -> int !input)` reads an integer from standard in
    - `print : (int -> unit !output)` writes an integer to standard out
    - `load : (int -> int !read)` reads an integer from the specified integer memory location 
    - `store : int -> (int -> unit !write)` writes an integer to the specified integer memory location
    
Below is a sample program, which takes in a number as input and computes its square:

```
x : int !input = scan ();
y : int = mul x x;
_ : unit !output = print y;
```

Notice the special assignment syntax: every expression must be assigned to a variable (with `_` as a dummy) and every effect of an assigned expression must be encoded exactly in the type annotation. These requirements, which are enforced by the typechecker, guard against user error: it is impossible to cause a side effect without knowing about it.

All functions are values, and use the lambda-calculus-inspired syntax `\x : t . s` in which `x` is the argument, `t` is its type, and `s` is a statement to execute. In addition, there is a short form `\x : t . e` where the statement is replaced by an expression to be immediately returned.

Below is a sample program demonstrating functions and their properties:

```
add5 : int -> int = add 5;
twice : (int -> int) -> int -> int = \f : int -> int. \x : int. f (f x);
print_thrice : (int -> unit !output) = \x : int. {
  _ : unit !output = print x;
  _ : unit !output = print x;
  _ : unit !output = print x;
}
_ : unit !output = print_thrice (twice add5 2);
```

# Implementation
The language is written entirely in OCaml, using Jane Street Core as standard library, ocamllex as lexer-generator, and Menhir as parser-generator. The provided implementation includes a typechecker, an interpreter, and a sample optimization pass. The `Or_error` monad and its corresponding `let%bind`/`let%map` monadic syntax is used throughout the implementation to capture and thread through errors.

## Typechecker
The typechecker takes a parsed program AST as input and checks the types. For example, the `Assign (x, t, effs, e)` rule determines the type of the right-hand-side expression and checks if it has type `t` and effects `e` before adding `x : t` to the type context. The `If (e, s1, s2)` rule ensures that `e` has type `bool`, and takes the intersection of the type contexts from `s1` and `s2`. Since all expressions must be bound by an assignment, and only `Apply` expressions can cause side effects, all effects must be captured by assignment statements. 

The standard library provides a mapping from function names to function types, which is used as the starting type context for typechecking. This had the unfortunate constraint of disallowing `print` from supporting a variable number of arguments; varargs support would require an extra wrapper around types in the type context or a new `Any` type, neither of which were appealing options for this project.

## Interpreter
The interpreter takes a parsed program AST as input and executes the program, relying on the typechecker to catch type errors which would crash the interpreter. For example, the interpreter will fail if it makes an ill-typed function call, but running the typechecker first would catch those calls. Starting from an empty map from variable names to their values, each statement of the list of top-level statements comprising the entire program is executed in order, with the resulting context being threaded from one statement to the next. In this implementation, effects are ignored at runtime (i.e., interpreter time), but runtime effect support is discussed as a potential extension.

The most difficult part of this module was correctly evaluating lambdas and closing around the variables needed, especially in partial application situations. To that end, `Lambda (x, t, s)` had to be extended in the code to `Lambda (x, t, ctx, s)`, with `ctx` representing the context of the closure upon creation. Then, partial application could be implemented as nested lambda expressions, and evaluating one layer of a lambda stack would encode its argument into the context of the next layer. 

## Optimizer
The optimizer included with the language is very simple, demonstrating only one of the possible optimizations. In essence, relying on the typechecker to properly enforce the effect annotations, the optimizer is able to collapse pure expressions that have fixed inputs, essentially performing a combination of inlining and constant propagation. This can be seen in the [factorial sample program](https://github.com/kq-li/type-and-effect-system/blob/master/test/basic/fact.txt), which automatically inserts the precomputed value of 5! after an optimization pass. A further dead code elimination pass, not implemented here, could drastically reduce program size in this case. 

For each of the small benchmark programs in the `test/basic/` subdirectory, an optimized version will run in the `test/optimize/` subdirectory. The comparison script `test/compare.sh` will diff the results from these two runs; at the time of writing, the results of this diff are as follows:
- `fact.txt` is improved from 24 executed statements to 4
- `func.txt` is improved from 8 executed statements to 6
- `func3.txt` is improved from 20 executed statements to 8
- all other program execution counts were unchanged

Though not deep, this level of empirical evaluation seemed appropriate considering the scope of the optimization: any memory manipulation, output, or input (e.g., variable input) would result in no optimizations at all.

## Standard Library
One interesting thing about the language is that the only primitive expressions are values and function applications. All operations on integers and booleans are part of the standard library. Encoded as `Extern` values, these functions are actually OCaml functions, made possible by the fact that the language uses an OCaml interpreter. This simplified implementation for this project, but if compilation to assembly is desired, the standard library operations would likely have to be implemented as primitives, especially for the sake of performance (so as not to incur function call overhead for each simple arithmetic operation).

The memory extension was also very simple to implement: the program memory is represented by a mutable array in the library module, and the `load` and `store` functions are just `get`s and `put`s on that underlying array. Again, this simplification is made possible by the fact that the language uses a high-level interpreter, but it is possible to translate this convenience implementation to a more performant and effective low-level program.

# Potential Extensions
One limitation of the strict assignment syntax is that writing code with many side effects can quickly become cumbersome. To alleviate the verbosity, a type inference algorithm could be used, allowing programmers to leave out type annotations where unneeded. The downside of such an approach would be a reduction in visibility of effects, which is one of the justifications for this sort of system; further investigation would help find a better balance between usability and correctness.

Additionally, proving the soundness of the type system was out of the scope of the project. However, the effects are simply lifted from expressions up to statements and function types, and the language itself bears a more-than-passing resemblance to IMP and the lambda calculus, so the typing rules shouldn't be too difficult to derive and prove sound. 

There are almost definitely still bugs in the interpreter, particularly relating to how contexts are passed around and changed; more careful thought is needed to fully formalize the semantics of the language.

The current optimizer is extremely limited in scope, since in practice code will generally have variable inputs. A more sophisticated optimizer could take advantage of the side effect information provided to reorder instructions, which could enable all sorts of optimizations. Another deficiency of the current optimizer is that it does not represent the instructions in an optimization-friendly way -- i.e., it doesn't process the program into basic blocks and a control flow graph, making it more difficult to reason about the context and variable liveness and other information flow through the program. Implementing such a framework would not be too difficult -- since the only control flow statements are `if`, `while`, and `return`, basic blocks would just be composed of all statements in the bodies and between these statements, and finding the edges of the CFG would be fairly straightforward. Then, more advanced optimization techniques like dataflow analysis and more could take further advantage of the refined type and effect information.

The original project proposal included the idea of runtime effect support and user-defined effects, which would allow for the implementation of resumable exceptions and generators. However, during the development process, it seemed more productive to focus on demonstrating more useful effects such as memory reads and writes. A further extension of the memory as used here would be to add arguments to the memory type annotations representing the region of memory affected, and using this region information to determine whether two memory instructions come into conflict or whether they can be reordered without issue.

# Conclusion
The language underwent many revisions; starting as a simple lambda calculus, then becoming more imperative, before shifting back with the reimplementation of higher-order functions and currying. This constant pivoting resulted in several large-scale reimplementations and refactors, which inhibited progress and ultimately resulted in a less fully-featured and tested project than originally desired. All that said, the end result is still fairly promising: the language is able to perform typical computations; the typechecker and interpreter seem correct and easily extensible; and even in the process of writing the benchmarks, the effect annotations prevented some bugs. 
