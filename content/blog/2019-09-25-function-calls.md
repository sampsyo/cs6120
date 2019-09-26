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

We extended the JSON representation of Bril functions to account for a function's arguments and return type. Every instruction still has a name and a list of instructions. 

```
{ "name": "<string>", "instrs": [<Instruction>, ...], "args": [<Argument>, ...], "type": <Type>}
```

A function can take no arguments, in which case the \"args\" field contains the empty list.
The return type, represented by the \"type\" field, is not required. A function that does not return anything (giving it the return type `void`) does not contain the \"type\" field.

An Argument JSON object contains the argument's name and type:

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

Manual testing uncovered several significant bugs via manual testing. 

When were fairly confident we had finished our implementation (hah), we wrote a quick recursive Factorial implemention in the TypeScript frontend:

```
function fac(x : number) : number {
    if (x <= 1) return 1;
    var result = x * fac(x - 1);
    return result; 
}
```

Surprisingly, this test failed to compile to Bril—we had forgotten that in TypeScript, function calls could be nested subexpressions! Our implementation expected functions that did not return void to be stored directly into variables. 

2. 'void' written explicitly as a function type in typescript
3. nondeterministic lark parsing of boolean variable declarations sometimes as value operations instead of constant operations

- couldn't cover type error messages in typescript because bools aren't properly compiled in `ts2bril.ts`.

### Automated property-based testing with [Hypothesis][]

We were excited to try our hand at stress testing our Bril implementation with automated testing. 
The key idea behind property-based testing is to specify some details of expected program behavior, then use an framework to test those ideas on many automated examples (in particular, more than a human would reasonably want to write). 

For Bril, we decided to use a python-based property testing framework, Hypothesis. 
The primary challenge in using such a tool is to specify _how_ example data can be generated such that the tests are useful. 
In testing Bril, this meant specifying how to generate syntactically correct Bril programs.

Our first test checks the property that conversion from text-based Bril JSON to is invertible. 
That is, we want the following high level assertion to hold:
```
bril2json(bril2txt(program)) == program
```

For this test, we don't particuarly care if the programs we generate are _meaningful_, as long as they are of the correct syntatic for. 
We can also generate the simplier, JSON syntactic form. 
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
    typ = draw(types)
    if (typ == "int"):
        value = draw(sampled_from(range(100)))
    elif (typ == "bool"):
        value = draw(sampled_from([True, False]))
    return {
        "op": "const",
        "value": value,
        "dest": draw(names),
        "type": draw(types)}
```
Here, we use a sampling primitive to choose either `int` or `bool`, then generate a numeric or boolean value as appropriate.

Along with similar composite strategies for other instruction forms (including calls) and functions, we build up many (somewhat silly) programs. Even this naive strategy found a potential bug:

```
{'functions': [{'args': [], 'instrs': [{'dest': 'aaa', 'op': 'const', 'type': 'bool', 'value': True}], 'name': 'main', 'type': 'int'}]}  !=
{'functions': [{'args': [], 'instrs': [{'dest': 'aaa', 'op': 'const', 'type': 'bool', 'value': 'true'}], 'name': 'main', 'type': 'int'}]}
```

Originally, we generated the JSON strings `true` and `false` (instead of boolean literals `True` and `False`). The `bril2txt` implementation parsed this correctly, which we decided to leave as-implemented, but this assured us that Hypothesis could actually find programs that were not reversible as we expected.

We also tested that running Hypothesis-generated programs through the `brili` Bril interpreter only produced clean-exit expected error cases, instead of exposing failures in the underlying TypeScript implementation. Once we changed `Brili` to throw a specific exception, this meant testing the high-level property:

```
exit_code = brili(program) 
exit_code == 0 || exit_code == <known exit code>
```

Because we did not encode much semantic meaning into the generation strategies, almost all generated programs failed in the interpreter (some did execute, and print values, sucessfully!). Reading the generated programs also led us to realize that we were not specifically handling the case where a Bril program calls a function with multiple definitions. 

Overall, property-based testing was easier than expected to set up, and helped us explore the sample space of Bril programs.

## Next steps

1. explicit program stack
2. first-order/anonymous functions
3. integrating with a static type checker, which would let us remove dynamic function call type checking
4. TS main arguments and return code












