+++
title = "Function calls in Bril"
extra.authors = { "Alexa VanHattum" = "https://cs.cornell.edu/~avh", "Gregory Yauney" = "https://cs.cornell.edu/~gyauney" }
extra.bio = """
  [Alexa VanHattum][] is a second-year student interested in the intersection of compilers and formal methods. She also enjoys feminist book clubs and cooking elaborate [fish truck][] meals.

  [Gregory Yauney][] is a second-year student working on machine learning and digital humanities.
  
[alexa vanhattum]: https://cs.cornell.edu/~avh
[gregory yauney]: https://cs.cornell.edu/~gyauney
[fish truck]: https://www.triphammermarketplace.com/events/
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
In service of [this course][]'s journey toward successively more fun compiler hacking, we set out to rectify this \"oversight\". 

The Bril ecosystem is centered around a JSON-based intermediate language that represents functions, labels, and instructions.
In addition, Bril includes two _frontends_ to make for a more ergonomic programming experience—users can compile from either a more concise text-based syntax or a restricted subset of TypeScript.
For our project, we decided to focus our scope on simple function calls (without first-class functions) in favor of updating the full Bril stack.

[adrian]: https://www.cs.cornell.edu/~asampson/
[this course]: https://www.cs.cornell.edu/courses/cs6120/2019fa/

## What we did

### Surface syntax

Bril now supports function definitions:
```
<ReturnType> <function name>(<argument name> : <Type>, ...) { <instructions> };
```

Where:
- `<ReturnType>`: The return type of a function must be `void` or one of the currently recognized Bril types: `int` or `bool`.
- `<function name>`: The function's name.
- `<argument name> : <Type>`: There can be zero or more arguments. Each argument name must be paired with a Bril type.
- `<instructions>`: This is a sequence of Bril instructions.

Bril now supports two kinds of `call`s, those that produce a value (value operation), and those that do not (effect operation):
```
var <variable name> : <Type>  = call <function name>(<args>);
call <function name>(<args>);
```

For backwards compatibility, functions can still be declared without return types and arguments, as in `tests/ts/br.bril`. 
Such functions are assumed to have a return type of void.

### Extended JSON representation

We extended the JSON representation of Bril functions to account for a function's arguments and return type. Every `Function` object still has a name and a list of instructions. 

```
{ "name": "<string>", "instrs": [<Instruction>, ...], "args": [<Argument>, ...], "type": <Type>}
```

A function can take no arguments, in which case the `\"args\"` field contains the empty list.
The return type, represented by the `\"type\"` field, is not required. A function that does not return anything (giving it the return type `void`) does not contain the `\"type\"` field.

An `Argument` JSON object contains the argument's name and type:

```
{"name": "<string>", "type": <Type>}
```

The JSON Bril `Program` object remains unchanged as a list of functions.

### Compiling to JSON

We extended the frontend for text-based Bril in `briltxt.py`.
The goal was to convert our new function definitions and call instructions to the JSON representation of Bril, necessitating extending the parser and JSON generators.

We also extended the TypeScript frontend in `ts2bril.ts`.
The TypeScript parser already handled calls to handle treating `console.log` statements as Bril's `print` statements. We extended this component to also capture effectful calls that return results. In addition, the initial implementation did not support function clarations, so we added new support to transform the declaration and type information.

### Interpreter

The interpreter needed to be able to handle functions and calls in their extended JSON representation. 
The main work was when encountering a `call` instruction: we create a new, empty environment with the arguments bound to the correct values.
The interpreter searches for the function name in the program's list of functions since we are not implementing first-class functions.
Because we chose to represent the stack implicitly, function calls are executed with a recursive call to `evalInstr`, thus relying on the underlying TypeScript stack frame implementation.

Helpful compilers also need to check for errors. The interpreter now checks for a number of possible errors when calling functions. We use simple dynamic type checking to ensure that (1) argument types match the types of the provided values and (2) the function's declared return type matches both the type of the returned value and the type of the variable where the returned value is being stored.

Below, we discuss our design decisions and their impact on our implementation.

### Design decisions

There were surprisingly many decisions to be made in the course of designing function calls.

- For the sake of a sufficiently-scoped project, we chose not to implement first-order functions.
- We implicitly represent the stack with recursive interpreter calls for simplicity based on the functionality we target.
An explicit stack would allow more interesting control flow in the future.
- We chose to allow backwards compatibility with the original Bril `main` syntax that did not have a return type or arguments.
Similarly, the typescript `main` function is not explicitly demarcated&mdash; it is understood to consist of the instructions before any function definitions.
- Calls can be effectful or non-effectful.
In the JSON representation of Bril, we chose to represent `call` as its own 'kind' of instruction, allowing us to include the function's name as an explicit `name` field in the JSON object rather than an argument to the instruction.
- If multiple functions with the same name are defined or a called function is missing, the interpreter throws an error.
- `main` functions in the text and JSON Bril representations can take arguments that are fed to `brili`. `main` also takes named and typed arguments, rather than C-style `argc/argv`.
However, `main` doesn't return an exit code for simplicity.
- Originally, the Bril interpreter simply threw string message exceptions on errors. We made the design decision that the interpreter should not leak interpretation details through uncaught excerptions for anticipated failures. We updated the interpreter to return a specific exception, which is caught and send to standard error along with a custom exit code.

### Challenges

The hardest part of this particular project, as with many compiler endeavors, was wrangling with new frameworks and existing code bases. 
In particular, this project was more involved than we originally expected because it touched the full Bril stack&mdash;not just the interpreter, but the text-to-JSON and JSON-to-text compilers, the TypeScript frontend, and the Turnt testing framework. 

The TypeScript frontend changes were especially gnarly because the TypeScript AST does not have detailed documentation. It took us quite some time to determine how to determine, for example, if a function call AST node stored its result to a variable. 

Finally, the Hypothesis testing framework was completely new for us, so it was somewhat challenging to think of how to generate meaningful test data automatically. In the end, we settled on generating relatively simple syntactically correct programs. It would be interesting to put more time into generating richer, semantically meaningful Bril in the future as well. 

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

### Automated property-based testing with [Hypothesis][]

We were excited to try our hand at stress testing our Bril implementation with automated testing. 
The key idea behind property-based testing is to specify some details of expected program behavior, then use a framework to test those properties on many automated examples (in particular, more than a human would reasonably want to write). 

For Bril, we decided to use a python-based property testing framework, Hypothesis. 
The primary challenge in using such a tool is to specify _how_ example data can be generated such that the tests are useful. 
In testing Bril, this meant specifying how to generate syntactically correct Bril programs.

Our first test checks the property that conversion from text-based Bril JSON to is invertible. 
That is, we want the following high level assertion to hold:

```
bril2json(bril2txt(program)) == program
```

For this test, we don't particularly care if the programs we generate are _meaningful_, as long as they are of the correct syntactic form. 
We can also generate the simpler, JSON syntactic form. 
In Hypothesis, this is accomplished via _strategies_ that tell the framework how to compose test data. 
We start with the small forms, and build up to a whole program.
For example, we can generate simple names with the following, which says that names are 1-3 lowercase Latin characters:
```
names = text(alphabet=characters(min_codepoint=97, max_codepoint=122), 
             min_size=1,
             max_size=3)
```

Instructions are built up compositionally, using a `draw` primitive that automatically explores the specified space of the constitiant parts. For example, constant instructions are generated with:

```
types = sampled_from(["int", "bool"])

@composite
def bril_constant_instr(draw):
    typee = draw(types)
    if (type == "int"):
        value = draw(sampled_from(range(100)))
    elif (type == "bool"):
        value = draw(sampled_from([True, False]))
    return {
        "op": "const",
        "value": value,
        "dest": draw(names),
        "type": type)}
```
Here, we use a sampling primitive to choose either `int` or `bool`, then generate a numeric or boolean value as appropriate.

Along with similar composite strategies for other instruction forms (including calls) and functions, we build up many (somewhat silly) programs. Even this naive strategy found a potential bug:

```
{'dest': 'aaa', 'op': 'const', 'type': 'bool', 'value': True} !=
{'dest': 'aaa', 'op': 'const', 'type': 'bool', 'value': 'true'}
```

Originally, we generated the JSON strings `true` and `false` (instead of boolean literals `True` and `False`). The `bril2txt` implementation parsed this correctly, which we decided to leave as-implemented, but this assured us that Hypothesis could actually find programs that were not reversible as we expected.

We also tested that running Hypothesis-generated programs through the `brili` Bril interpreter only produced clean-exit expected error cases, instead of exposing failures in the underlying TypeScript implementation. Once we changed `Brili` to throw a specific exception, this meant testing the high-level property:

```
exit_code = brili(program) 
exit_code == 0 || exit_code == <known exit code>
```

Because we did not encode much semantic meaning into the generation strategies, almost of the all of the thousands of generated programs failed in the interpreter (some did execute, and print values, successfully!). Reading the generated programs also led us to realize that we were not specifically handling the case where a Bril program calls a function with multiple definitions. 

Overall, property-based testing was easier than expected to set up, and helped us explore the sample space of Bril programs.

## Next steps

There are several interesting directions that Bril's function handling could take from here. We could represent the program stack and context explicitly, rather than relying on the underlying interpreter's stack, and implement first-order and anonymous functions. We could also integrate with other projects' type checking and eliminate most of the interpreter's static checks. Finally, function calls will allow us and other Bril implementors to run more exciting programs (and optimizations) in the future.
