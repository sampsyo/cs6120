+++
title = "Type Inference for Bril"
extra.bio = """
  Christopher Roman is a second semester MEng student in computer science. He is interested in compilers and distributed systems. He is also the best [Melee](https://en.wikipedia.org/wiki/Super_Smash_Bros._Melee) player at Cornell. :)
"""
[[extra.authors]]
name = "Christopher Roman"
+++

## Overview
The goal of this project is to build a type inferrer for Bril. Ultimately, we want to
take a Bril program that has some (or none) of the types specified on variables
and produce a new Bril program that has all the correct type annotations.
For example, this program:
```
main {
  v0 = const 1;
  v1 = const 2;
  v2 = add v0 v1;
  print v2;
}
```
should be transformed into the program:
```
main {
  v0: int = const 1;
  v1: int = const 2;
  v2: int = add v0 v1;
  print v2;
}
```
This is useful in cases where the frontend isn't typed. For example, someone
building a Python frontend would need to specify the types of variables, which
may be annoying. This takes some burden off the frontend implementation.
Additionally, it makes it quicker to write Bril programs because we don't have
to worry about correctly typing the variables.

We also will implement type checking through this type inference.

## Design
For our workflow, we'd like to take a semi-typed bril program and convert it
between JSON and text representations. This requires changing the grammar so
that type annotations are optional.

Because Bril only supports two types, `int` and `bool`, the type inference is
very straight forward. For example, consider arithmetic operations, i.e.
`add`, `mul`, `sub`, and `div`. The arguments to this operations *must* be ints,
and the result is an int. Therefore, if we see a statement like `x = add a b`,
we know `x`, `a`, and `b` are all ints. If at any point we inferred that one
of these variables are not ints, then we have a type unification error, and the
program is not well-typed.

## Implementation

### Modifying the Grammar
We want to make type annotations optional. So first I modified the grammar to
have a rule for type annotations:
```
type_decl.5: ":" type
const.4: IDENT [type_decl] "=" "const" lit ";"
...
```
This is an issue however because there is an ambiguity between labels and
assignments. Consider these two Bril programs:
```
main {
l:
  x = const 5;
}
```
and
```
main {
  l: x = const 5;
}
```
These programs consist of the same tokens. However, the first one intuitively
means that we have a label `l` and some int `x` with a value of 5. The second
one means that we have a variable called `l` of type `x` with a value of 5.
It is incorrect to simply have one rule as a higher priority, because
semantically, both of these should be allowed as separate programs. So we have
two options:
1. Force labels to be on its own line
2. Only allow fixed type names, like "int" and "bool"

I decided to go with (2) because I didn't know how to easily do (1), but in
retrospect (1) may have been better, because some other students' projects
allow for user-defined types.

### Type Inference
As noted in the design, inferring the types of variables for a single statement
is straightforward. Here is an example snippet showing how types for comparison
ops are inferred:
```
...
elif instr["op"] in COMPARISON_OPS:
    for arg in instr["args"]:
        type_var(gamma, arg, "int", i)
    type_var(gamma, instr["dest"], "bool", i)
...
```
Here, we keep track of a typing context `gamma` which maps variables to their
type. Then, we check that each argument is either untyped or already has the
type `int`; otherwise we'll throw a type unification error. Finally, we do
the same for the destination, making sure that it is typed as `bool`.

For our implementation, we simply iterate through each instruction and determine
the types of our variables. The one case where this presents issues is with
`id`. In general, `id` sets the type of the destination to the type of the
variable on the right hand side. However, what happens if we don't know the type
of the variable being copied? For example, consider this Bril program:
```
main {
  jmp later;
earlier:
  x = id y;
  ret;
later:
  y = const 5;
  jmp earlier;
}
```
This is a valid program that should typecheck. Specifically, `x` and `y` are
both ints. However, if we naively go sequentially through each instruction, we
don't know the type of `y` until we have `y = const 5`, at which point it is too
late to type `x`. To resolve this, we have two options:
1. Rerun the type inference algorithm, stopping when no more variables have been
inferred
2. Keep track of which variables must have the same type, and after 1 pass,
setting their types to be the same.

For simplicity sake, we choose (1). This means in the worst case, type inference
will take `O(n^2)` time, where `n` is the number of instructions. This happens
when there are multiple such `id` assignments as in the example above.

### Type Checking
With type inference, type checking is relatively simple. After type inference,
we have the original Bril program and the fully typed Bril program. We then go
through the original Bril program and make sure that for any variable that has a
type annotation, the type matches the inferred type. For completeness, I also
check to make sure that variables aren't being used as labels, and vice versa.

## Evaluation
To properly evaluate that everything presented here is correct, we have to test
the parser and the type inferrer. To do these simultaneously, we can run tests
as follows:
```
command = "cat {filename} | bril2json | python ../../infer.py | bril2txt"
```
We are taking Bril text programs, turning them into JSON, generating a new
equivalent typed program, and turning it back to text. First I started with
simple Bril programs that already existed to build some confidence. Then to gain
full confidence, I wrote tests that use every kind of operations, e.g.
arithmetic ops, comparison ops, logical ops, effect ops, and misc. ops.

I tested for both positive and negative results. In other words, I ensured
Bril programs that *should* typecheck were correctly type-inferred, and programs
that *shouldn't* typecheck in fact could not be type-inferred. For example:
```
main {
  F1 = const false;
  T1 = const 1;
  b1 = and F1 T1;
}
```
Unfortunately I couldn't get Turnt to validate the error message that was
output, so instead I manually checked to see that the error was what I expected.
The exception raised for the above program is:
```
Exception: (stmt 3) Expected "T1" to have type "bool" but found "int"
```

I also made sure to test Bril programs that contained some type annotations and
made sure that didn't interfere with the inference. This same process was
repeated for testing the typechecker, which was run by passing the `-t` flag to
`infer.py`:
```
command = "cat {filename} | bril2json | python ../../infer.py -t | bril2txt"
```

Through this process, I uncovered some bugs when implementing the type inferrer:
- grammar ambiguity, detailed in the "Modifying the Grammar" section
- wrongly tested for literals True and False using `==` instead of `is`
- accidentally had comparison ops' args be bools

By rigorously testing all possible language features, we can be confident that
the type inference is correct.

## Hardest Parts to Get Right
The hardest parts to get right were the parser and full correctness of the type
inferrer. I didn't realize the issue with the parser towards the beginning
because I arbitrarily set a priority for the type annotation rule; debugging
this took a while. Additionally, it was really easy to get a type inferrer
that seemed to work for majority of cases. However, small bugs like the ones
mentioned previously were only fixed through intense testing. Comparatively, the
actual type inference and typechecking was relatively straightforward, mostly
because there are only two types in Bril and no function calls.

## Possible Extensions
Other groups are working on adding function calls to the language. By doing
type inference on the function arguments and checking the return type, we can
determine the type of a particular function. From this, we can relatively easily
infer types for assignments to function calls.

