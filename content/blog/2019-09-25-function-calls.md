+++
title = "Function calls in Bril"
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

### Surface syntax for calls and fully-fledged function definitions in Bril and typescript

Bril now supports function definitions:
```
<ReturnType> <name>(<arg_1> : <type_1>, ..., <arg_n> : <type_n>) { <instructions> };
```

Where:
- `<ReturnType>`: The return type of a function must be `void` or one of the currently recognized Bril types: `int` or `bool`.
- `<name>`: The function's name is a string that can consist of letters, numbers, and underscores. It cannot begin with a number.
- `<arg_i> : <type_i>`: Each argument name must be paired with a Bril type.
- `<instructions>`: This is a sequence of Bril instructions.

Bril now supports two kinds of `call`s, those that produce a value (value operation), and those that do not (effect operation):
```
var <name> : <type>  = call <name>(<args>);
call <name>(<args>);
```

For backwards compatibility, functions can still be declared without return types and arguments, as in `tests/ts/br.bril`. 
Such functions are assumed to have a return type of void.

### Extended JSON representation

We extended the JSON representation of Bril functions to account for a function's arguments and return type. Every `Function` object still has a name and a list of instructions. 

```
{ "name": "<string>", "instrs": [<Instruction>, ...], "args": [<Argument>, ...], "type": <Type>}
```

A function can take no arguments, in which case the \"args\" field contains the empty list.
The return type, represented by the \"type\" field, is not required. A function that does not return anything (giving it the return type `void`) does not contain the \"type\" field.

An `Argument` JSON object contains the argument's name and type:

```
{"name": "<string>", "type": <Type>}
```



The JSON Bril program object remains unchanged as a list of functions.

### Compile to JSON (Bril IR)

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
Our evaluation was two-fold: (1) manual testing at multiple abstraction levels (JSON, text-based Bril, and TypeScript), and (2) automated property-based testing to try and cover classes of errors we may not have anticipated. In order to support these lofting testings goals, we also had to make several tooling changes.

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

Manual testing uncovered several significant bugs. 

When we were fairly confident we had finished our implementation (hah), we wrote a quick recursive factorial implemention in the TypeScript frontend:

```
function fac(x : number) : number {
    if (x <= 1) return 1;
    var result = x * fac(x - 1);
    return result; 
}
```

Surprisingly, this test failed - we had forgotten that in TypeScript, function calls could be nested subexpressions! Our implementation expected functions that did not return void to be stored directly into variables. We did not have to worry about this in text-based Bril because operations can only take variables as their arguments.

Testing `void` functions revealed that the TypeScript compiler was expecting only annotated function types of `number` and `boolean`.
Though the legacy syntax for defining a `void` function&mdash;without any type annotation&mdash;compiled fine, the test showed that we had to add a check for an explicit `void` return type.

We also found a bug arising from the nondeterminism of Lark, the Python parser; constant operations were occasionally parsed as value operations. This was fixed with a simple upgrade to the most recent version of Lark.

Finally, we found a bug in the original TypeScript compiler (`ts2bril.ts`) while manually testing the argument type error messages of our function implementation (TODO link Bril issue).
Hopefully we can fix this soon!
The compiler hits an unexpected error when encountering a boolean variable declaration (with or without the type annotation):

```
var x : boolean = true;
```

### Automated property-based testing with Hypothesis

1. no error-checking of "true" vs. True in `bril2txt`
2. reading generated programs made us realize we don't check for function name collisions (hopefully, can hit this!)

## Next steps

1. explicit program stack
2. first-order/anonymous functions
3. integrating with a static type checker, which would let us remove dynamic function call type checking
4. TS main arguments and return code












