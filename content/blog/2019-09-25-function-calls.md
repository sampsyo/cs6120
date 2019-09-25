+++
title = "Function calls in BRIL"
extra.authors = { "Alexa VanHattum" = "https://cs.cornell.edu/~avh", "Gregory Yauney" = "" }
extra.bio = """
  [Alexa VanHattum](https://cs.cornell.edu/~avh) [[TODO]]

  [Gregory Yauney](https://cs.cornell.edu/~gyauney) [[TODO]]
"""
+++


//## Goal: Function calls in BRIL

## Function calls in Bril
In this post, we will describe our experience extending Bril (the Big Red Intermediate Language) to include function calls. In addition, we share how we tested our implementation with both targeted manual tests and automated property-based testing (a la QuickCheck) with Hypothesis.

### Why did Bril need function calls?

Bril is a simple, extensible language that @sampsyo designed to be a playground for building compiler extensions and optimizations. While out-of-the-box Bril supports programs with multiple functions, the initial implementation lacked an instruction to actually _call_ one function from another. In service of this course's journey toward successively more fun compiler optimizations, we set out to rectify this gap by implementing function calls. 

The Bril ecosystem is centered around a JSON-based intermediate language to represent functions, labels, and instructions. In addition, Bril includes two _front-ends_ to make for a more ergonomic programming experience---users can compile from either a concise text-based syntax or a restricted subset of TypeScript. For our project, we decided to focus our scope on simple function calls (without first-class functions) in favor of updating the full Bril stack.

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

For backwards compatibility, functions can still be declared without return types and arguments, as in `br.bril`. Such functions are assumed to have a return type of void.



### Compile to JSON (BRIL IR)

### Interpretation (in `brili.ts`)

### Extended turnt to test for expected errors

- necessitated adding comments in BRIL

### Automated property-based testing with Hypothesis

### Design decisions

1. implicitly represent stack with recursive interpreter calls
2. no first-order functions
3. backwards compatibility
	- typescript implicit main
4. calls can be effectful or non-effectful (call is its own 'kind' of instruction)
5. TODO: multiple functions
6. arguments for main: feed to brili (what adrian said, no argv/-c)
	- main doesn't return an exit code

### Hardest parts

0. touching the whole stack (both frontends, json representation, interpreter, test framework)
1. typescript ast
2. generating reasonable programs in hypothesis

## Evaluation

### Description of new tests

- couldn't cover type error messages in typescript because bools aren't properly compiled in `ts2bril.ts`.

### Errors found by tests

1. call as nested subexpression (found by recursive factorial)
2. 'void' written explicitly as a function type in typescript
3. nondeterministic lark parsing of boolean variable declarations sometimes as value operations instead of constant operations

### Hypothesis

1. no error-checking of "true" vs. True in `bril2txt`
2. reading generated programs made us realize we don't check for function name collisions (hopefully, can hit this!)

## Next steps

1. explicit program stack
2. first-order/anonymous functions
3. integrating with a static type checker, which would let us remove dynamic function call type checking
4. TS main arguments and return code












