+++
title = "An LLVM Frontend for a Small Programming Language"
[[extra.authors]]
name = "Shubham Chaudhary"
link = "https://www.cs.cornell.edu/~shubham/"
[[extra.authors]]
name = "Anshuman Mohan"
link = "https://www.cs.cornell.edu/~amohan/"
[[extra.authors]]
name = "Ayaka Yorihiro"
link = "https://ayakayorihiro.github.io"
+++

# Introduction

We report on `BRAWN`, the Big Red AWk implementatioN. Our project is in service of `AWK`, which is a scripting language used to process text in a filter style. We build a new `LLVM` frontend for `AWK`, and then leverage `LLVM`'s backend to generate optimized machine code. The standard implemetation of `AWK`, GNU `AWK`, is interpreted, while `BRAWN` is compiled.

### Where we are

We compile a `.brawn` file, which supports almost all of `AWK`'s features, to `LLVM` IR. We then
link this to a runtime which implements all the built-in operations and has a garbage collector
(Boehm's [^8]) to produce a working executable. Unfortunately we have not been able to test our
implementation extensively, except for a few test programs that excercise the basic machinery of
`BRAWN` (the `main` loop, arrays, loops, conditionals, printing, and regular expression matching).
Our code is [here](https://github.com/ayakayorihiro/cs6120-project).

# Implementation

## Overview
Here we skate over the parts of our implementation that are standard. We discuss more interesting issues in dedicated subsections.

### Our plan is fairly standard:
1. Parse `AWK` into `OCaml`.

2. Write a `C++` runtime module that provides the built-in operations that `AWK` promises to support but which are not standard in LLVM. Compile this module to `LLVM` IR.

3. From the `AWK` AST in `OCaml`, emit `LLVM` IR code. This code contains `extern` calls to the built-in operations.

4. Linking the two pieces above gives us `LLVM` IR code without gaps. Optimize this in `LLVM`.



### Here's how we do it:
1. We use `Menhir` [^3], which takes a grammar in the `yacc` format and generates a parser for `OCaml`.  We choose `Menhir`, and therefore `OCaml`, because one of the officially-released grammars [^4] for `AWK` is written in `yacc`.

2. The mechanics of this step are standard: just ask `clang++` to emit LLVM. This is also where we run into many gotchas and make many of our design decisions. We discuss these in future sections.

3. This work is in `OCaml`. We walk over the AST generated previously and define an `LLVM` IR production for each constructor, as suggested by the Kaleidocope guide [^1]. Loosely speaking, there are three kinds of tasks:
    * Issue straightforward calls to the `OCaml`-to-`LLVM` module [^2].
    * Issue `extern` calls to the runtime module created previously.
    * Make basic blocks and design control flow that is accurate to `AWK`'s behavior.


## Curiosities of `AWK`
1. `AWK` is dynamically typed. To deal with this, we introduce a type `brawn_value` which is a tagged union of a `double`, `std::string`, and a `std::unordered_map`. Our runtime implements built-in functions on these values.

2. `AWK` is typically interpreted and not compiled. With the technique outlined above, we find it straightforward to perform compilation since we no longer have to worry about types statically.

3. Certain built-in functions can be called with variable number of arguments, and they have somewhat unusual behaviour when this happens. For example, `length ()` succeeds: it gives the length of the argument `$0`. We deal with this at the codegen phase: we pattern-match against these cases and call the functions with the appropriate arguments.

4. User-defined functions can be called with fewer arguments than they require. The variables that do not have arguments at call time are given default values.

5. A `REGEX` literal appearing to the right-hand side of a match expression or in the arguments to `sub`, `gsub`, etc. is interpreted as a `REGEX` literal, as expected. In all other places it is interpreted as the *truth value* of the expression `$0 ~ REGEX`.

6. Supporting `AWK`'s syntax of `BEGIN`/`END` blocks requires custom control flow. The same is true of the `next` and `exit` commands. We implement these using C++ exceptions. When the compiler encounters a `next` or `exit` call, it calls the corresponding function in the runtime which throws an exception, thus trigerring the desired control flow in the main loop.

## Points of Divergence (BRAWN vs AWK)
For convenience, we restrict ourselves to a (large) subset of `AWK` with a few changes for ease of implementation. We find these choices defensible because they reduce burden on the parser and runtime, which while interesting, are not the focus of our project. Here are the points of divergence:

1. We restrict I/O functionality. We support only the `print` function (no `printf` or `sprintf`). We only support reading input from `stdin` and writing output to `stdout`. Hence, we do not support built-in variables like `FNR`, `CONVFMT`, `OFMT`, and `FILENAME`.

4. We require that all statement blocks like `if`, `for` etc. be followed by statements enclosed in `{}`.

6. `AWK` uses the token `/` to indicate both `div` and the beginning/end of a regular expression. `BRAWN` requires that regular expressions begin with `/#` and end with `#/`.

7. `AWK` offers a `bool`-returning membership query of the form `bird in birds`. `BRAWN` requires a different style: `[bird] in birds`.

8. `AWK` offers string concatenation via the syntax `"concat" "these" "strings"`. `BRAWN` requires `"concat" @ "these" @ "strings"`.

9. `AWK` allows functions to be declared anywhere in the program so long as they are outside of blocks. They can be called before they are declared. `BRAWN` requires that all functions be defined before any blocks are.


# Evaluation

## Correctness
For a small selection of `AWK` programs that we can support in `BRAWN`, we compare our outputs to that of the standard `AWK` implementation. We select a series of `AWK` programs, generally a little more complex than simple one-line commands, from benchmarks previously used by
* the `AWK`-like language [frawk](https://github.com/ezrosent/frawk)
* a [user guide](https://www.math.utah.edu/docs/info/gawk_toc.html) for GNU `AWK`
and convert them by hand into their equivalent `BRAWN` programs. The changes required are minor and are as described above. 

The programs perform the following operations:

| Name | Description |
|------|-------------|
| `math` [^6] | Compute means after grouping by the value of a column |
| `dupword` [^5] | Detect consecutive uses of words in a text |
| `word-count` [^5] | Report on the frequency of each word in a text |

We use `turnt` to generate the outputs of `AWK`, and compare these with `BRAWN` outputs by eye. Our compiled programs pass these tests.

## Benchmark
We provide a brief benchmark against the standard GNU `AWK` implementation. Because startup times are high, only longer-running programs offer interesting data. In our case, that is `math.awk` versus `math.brawn`. We run our benchmarks on a MacBook Pro running MacOS 12; the `AWK` version is 5.1.1.

Averaging over 100 runs of `math.awk`, a run takes 75 milliseconds. Averaging over 5 runs of `math.brawn`, a run takes **4.845 seconds**. For obvious reasons, we do not run the the latter test 100 times. This represents a _65x_ slowdown. This is disappointing, but we think some of the fault lies with `C++`'s implementation of `std::regex`, which is known to be slow [^7].


# Learnings

Much of the above took a lot of experimentation to figure out. Here are some of our primary takeaways from the project.

1. `AWK` is a strange language, with several instances of relatively nonstandard behavior in case of unexpected use. These corner cases do not bother the average good-faith user, but as writers of a compiler, we need to think exhaustively about such cases.
2. Parsing is hard! Even with a formal specification in `yacc`, we run into many a hurdle, and eventually bail ourselves out via opinionated choices in the grammar.
3. We should have taken Adrian's advice and written an interprefer at the `C++` level. This would have revealed many bugs in our runtime independent of end-to-end testing. 
4. `LLVM` is pretty fun! We benefit greatly from the abstractions and utilities that it offers, e.g. we can defer to an opaque `builder` to figure out much of the nitty-gritty of generating code. Doing this work in `BRIL` would have involved basic blocks, temporaries, and manner of control flow, and would, frankly, have been completely overwhelming.
5. That said, `LLVM` has its own pitfalls. We had a very frustrating time debugging an error that made `LLVM` crash with a `SIGILL` instead of a more informative error message. We reported this to Adrian, who told us that the release of `LLVM` does away with otherwise helpful asserts that would have proved more helpful. Using a debug version would have helped.


# References

[^8]: Boehm's GC: [https://www.hboehm.info/gc/](https://www.hboehm.info/gc/)

[^3]: Menhir: [http://gallium.inria.fr/~fpottier/menhir/](http://gallium.inria.fr/~fpottier/menhir/)

[^4]: The AWK specification: [https://pubs.opengroup.org/onlinepubs/009604499/utilities/awk.html](https://pubs.opengroup.org/onlinepubs/009604499/utilities/awk.html)

[^1]: The official guide to implementing a language using LLVM: [https://releases.llvm.org/8.0.0/docs/tutorial/OCamlLangImpl1.html](https://releases.llvm.org/8.0.0/docs/tutorial/OCamlLangImpl1.html)

[^2]: An OCaml module for generating LLVM IR: [https://llvm.moe/](https://llvm.moe/)

[^6]: Benchmark `math` obtained from [https://github.com/ezrosent/frawk/blob/master/info/performance.md#group-by-key](https://github.com/ezrosent/frawk/blob/master/info/performance.md#group-by-key)

[^5]: Benchmarks `word-count` and `dupword` obtained from [https://www.math.utah.edu/docs/info/gawk_toc.html#SEC154](https://www.math.utah.edu/docs/info/gawk_toc.html#SEC154)

[^7]: C++ and its slow regex: [https://stackoverflow.com/questions/41481811/why-pcre-regex-is-much-faster-than-c11-regex](https://stackoverflow.com/questions/41481811/why-pcre-regex-is-much-faster-than-c11-regex)

