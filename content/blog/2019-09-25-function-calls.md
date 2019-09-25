+++
title = "Function calls in BRIL"
extra.authors = { "Alexa VanHattum" = "https://cs.cornell.edu/~avh", "Gregory Yauney" = "" }
extra.bio = """
  [Alexa VanHattum](https://cs.cornell.edu/~avh) [[TODO]]

  [Gregory Yauney](https://cs.cornell.edu/~gyauney) [[TODO]]
"""
+++

## Function calls in Bril
In this post, we will describe our experience extending [Bril][] (the Big Red Intermediate Language) to include function calls. 
In addition, we share how we tested our implementation with both targeted manual tests and automated property-based testing (a la [QuickCheck][]) with [Hypothesis][].

[bril]: https://github.com/sampsyo/bril/blob/master/README.md
[quickcheck]: http://hackage.haskell.org/package/QuickCheck
[hypothesis]: https://hypothesis.works

### Why did Bril need function calls?

Bril is a simple language that [Adrian][] designed to be a playground for building compiler extensions and optimizations. 
While out-of-the-box Bril supports programs with multiple functions, the initial implementation lacked an instruction to actually _call_ one function from another. 
In service of this course's journey toward successively more fun compiler hacking, we set out to rectify this \"oversight\". 

The Bril ecosystem is centered around a JSON-based intermediate language that represents functions, labels, and instructions.
In addition, Bril includes two _front-ends_ to make for a more ergonomic programming experience—users can compile from either a more concise text-based syntax or a restricted subset of TypeScript.
For our project, we decided to focus our scope on simple function calls (without first-class functions) in favor of updating the full Bril stack.

[adrian]: https://www.cs.cornell.edu/~asampson/

## What we did

### Surface syntax for calls and fully-fledged function definitions in BRIL and typescript

BRIL now supports function definitions:
```
<return type> <name>(<arg1> : <type1>, ..., <argn> : <typen>) { <instructions> };
````

BRIL now supports both effectful call and value call instructions:
```
call <name>(<args>);
var <name> : <type>  = call <name>(<args>);
```

For backwards compatibility, functions can still be declared without return types and arguments, as in `br.bril`. 
Such functions are assumed to have a return type of void.

### Compile to JSON (BRIL IR)

### Interpretation (in `brili.ts`)

### Design decisions

1. implicitly represent stack with recursive interpreter calls
2. no first-order functions
3. backwards compatibility
	- typescript implicit main
4. calls can be effectful or non-effectful (call is its own 'kind' of instruction)
5. TODO: multiple functions
6. arguments for main: feed to brili (what adrian said, no argv/-c)
	- main doesn't return an exit code
7. Interpreter should not fail with implementation-specific errors (added custom exceptions)

### Hardest parts

0. touching the whole stack (both frontends, json representation, interpreter, test framework)
1. typescript ast
2. generating reasonable programs in hypothesis

## Evaluating our contribution

To convince ourselves that we'd actually made a useful contribution to Bril, we wanted to rigorously test our changes. 
Our evaluation was two-fold: (1) manual testing at multiple abstraction levels (JSON, text-based Bril, and TypeScript), and (2) automated property-based testing to try and cover classes of errors we may not have anticipated. In order to support these lofting testings goals, we also have to make several tooling changes.

### Tooling changes for testing

As we developed our implementation, we built up a  bevy of small Bril programs that we expected to trigger certain classes of errors.
However, the check-expect-style testing framework Bril employs, [Turnt][], did not support tests that were expected to fail.
We [extended][] Turnt to check both standard error and program exit codes in order to test invalid Bril programs. 

Turnt relies on C-style comments to configure settings on a per-test basis, so we also extended the Bril text-baseded surface syntax to support comments of the form `\\ <comment>`. 

Finally, in order for automated testing to be useful, we needed to distinguish between expected errors on invalid Bril programs and implementation flaws. 
We thus added a named exception to Bril's interpreter with an custom exit code, removing all string-based `throw` calls.

[turnt]: https://github.com/cucapra/turnt 
[functionality]: https://github.com/cucapra/turnt/issues/6

### Bugs we found with manual testing

We found several significant bugs via manual testing. 

1. call as nested subexpression (found by recursive factorial)
2. 'void' written explicitly as a function type in typescript
3. nondeterministic lark parsing of boolean variable declarations sometimes as value operations instead of constant operations

- couldn't cover type error messages in typescript because bools aren't properly compiled in `ts2bril.ts`.

### Automated property-based testing with Hypothesis

1. no error-checking of "true" vs. True in `bril2txt`
2. reading generated programs made us realize we don't check for function name collisions (hopefully, can hit this!)

## Next steps

1. explicit program stack
2. first-order/anonymous functions
3. integrating with a static type checker, which would let us remove dynamic function call type checking
4. TS main arguments and return code












