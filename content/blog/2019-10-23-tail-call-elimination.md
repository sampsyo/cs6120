+++
title = "Type Inference for Bril"
[extra]
bio = """
  Christopher Roman is a second semester MEng student in computer science. He is interested in compilers and distributed systems. He is also the best [Melee](https://en.wikipedia.org/wiki/Super_Smash_Bros._Melee) player at Cornell. :)
"""
[[extra.authors]]
name = "Christopher Roman"
+++

## Overview
The goal of this project is to implement tail call elimination in Bril. A tail
call is a call to a function whose value is immediately returned. For example,
`return foo()` is a tail call. Whenever we have a tail call in our Bril program,
we do not need to create a new stack frame for it. We can simply "overwrite"
the current stack frame since won't need any of the values in the frame anymore.
This is crucial for programming languages that use *tail recursion* as a
programming idiom (e.g., OCaml, Haskell, etc.). Consider the following (somewhat
contrived) TypeScript program:
```
function loop(n: number) {
  if (n == 0) {
    return 0;
  } else {
    return loop(n-1);
  }
}
```
This function simply loops `n` times, but does so in a functional way. Without
tail call elimination, programs like this would stack overflow with large values
of `n`. For languages that depend on this idiom, this is unacceptable.

## Design
The base Bril language is not rich enough to express tail call elimination, or
even function calls in general. To enrich the language, we can make Bril more
closely resemble something like x86 assembly. Intuitively, all Bril variables
implictly live on the stack. To pass arguments in a function call, we must
explicitly push them onto the stack. This is akin to pushing values on the stack
in assembly. When a function returns, those arguments are implictly popped off the stack.
Additionally, the return value of the function is obtained using a special
keyword that is akin to getting the return value from `rax` according to the
[System V Calling Conventions](https://en.wikipedia.org/wiki/X86_calling_conventions#x86-64_calling_conventions).

Importantly, we need the capability to jump to other functions, rather than just
labels in the current function. This way, if we have a tail call, then we can jump to
the beginning of the callee instead of creating a new stack frame. This is [how
tail call elimination is implemented](https://www.cs.cornell.edu/courses/cs4120/2019sp/lectures/34functional/lec34-sp19.pdf?1556325712)
in x86 assembly.

We have to modify the grammar to support features like pushing onto the stack
and defining functions, then update the interpreter. Thanks to the work done by
Alexa and Greg for [Project 1](https://github.com/sampsyo/bril/pull/16), these
changes were much easier for me to make.

## Implementation

### Modifying the Grammar
The first step is to modify the grammar to support function declarations with
arguments and return types, as well as support the various new value/effect
operations. Functions look as follows:
```
int foo(x: int, b: bool) {
  print x;
  print b;
  ...
}
```
Functions must specify a return type, a name, and a potentially empty list of
typed arguments.

Here are the new effect operations and their semantics:
- `push arg1 ... argn`: push arguments to the stack, which can be used by the
first function to be `call`'d after this instruction.
- `call foo`: starts executing instructions defined by the function `foo`.

Here are the new value operations and their semantics:
- `retval`: Retrieve the return value of the previous function call, e.g.
```
...
call foo
r: int = retval;
```
Here, if `foo` returned 0, then `r` would have the value 0.

### Extending the Interpreter
The interpreter needed to be extended to implement the above semantics. To model
a stack frame, I explicitly keep track of the program counter (i.e., which
instruction is being interpreted), the name of the current function, and an
environment. I made this explicit because it made jumping to other functions
easier to implement. The interpreter simply tries to evaluate the frame at the
top of the stack until the stack is empty.

Arguments that are `push`ed stay on the current stack frame. Once a `call` is
made, we create a new stack frame. Note that the names of the `push`ed arguments
don't necessarily match those declared by the function, so we need to map the
function's arguments to the values of the `push`ed arguments.

To return a value, a special variable name is set in the environment in the
previous stack frame so it can be retrieved by the caller using `retval`.

### Extending the TypeScript Frontend
For the majority of this, I referred to [Alexa and Greg's implementation](https://github.com/sampsyo/bril/pull/16).
There were some differences however because I have separate instructions for
passing arguments to a function. So, whenever a function call was found in the
AST, the arguments needed to be converted to Bril instructions first, then those
would be used in a `push`. Additionally, a `retval` would need to be created
afterwards if the result of the function call would be used.

### Identifying and Eliminating Tail Calls
The simple definition of a *tail call* would be an immediate return of a call
to a function. The translation from the TypeScript frontend of something like
```
return foo(n)
```
to Bril would be
```
push n
call foo
v: int = retval;
ret v
```
Thus we just need to look for `call`s that are immediately and optionally
followed by `retval`, and immediately followed by a `ret`.

This doesn't take into account more complex cases where there isn't an explicit
return of a function call, but the value returned comes from a call to the
same function from different branches. For example:
```
function foo(n: number): number {
  ...
  if (b) {
    result = foo(n-1);
  } else {
    result = foo(n-2);
  }
  return result;
}
```

To do this, we first do a global copy propagation. The dataflow analysis for
copy propagation that I used can be found [here](http://www.csd.uwo.ca/~moreno/CS447/Lectures/CodeOptimization.html/node8.html).
Then, for a value `v` that is `return`ed, we search backwards through the CFG until we find a `retval` that
corresponds to `v`, and make sure that it has not been modified along any of
these backwards paths, and the `reval` comes from a call to the same function.
In the above example, the corresponding Bril code looks something as follows:
```
then.6:
  ...
  call foo;
  v17: int = retval ;
  result: int = id v17;
  jmp endif.6;
else.6:
  ...
  call foo;
  v21: int = retval ;
  result: int = id v21;
endif.6:
  v22: int = id result;
  ret v22;
```
After copy propagation, it looks like this:
```
then.6:
  ...
  call foo;
  result: int = retval;
  jmp endif.6;
else.6:
  ...
  call foo;
  result: int = retval;
endif.6:
  ret result;
```

Then we can analyze the CFG backwards to see that indeed we can replace
the calls with `jmp` instructions.
```
then.6:
  ...
  jmp foo;
  jmp endif.6;
else.6:
  ...
  jmp foo;
endif.6:
  ret result;
```
Note that the extra instructions can simply be removed by a DCE pass, so we don't
worry about that.

*Unfortunately, I couldn't get this to work properly because my copy propagation
pass had bugs.*

## Evaluation
To evaluate that the tail call elimination is working and actually gives us an
improvement, we benchmark some recursive functions that use tail recursion, and
show the difference in execution time and memory usage between an optimized and
unoptimized Bril program.

The table entries show how much change was observed, as a percentage, by doing
tail call elimination (TCE). For example, an entry of -10% means the optimized program
used 10% less memory/time than the unoptimized program. An `X` means that the output of the program was too
big to handle. `n` is the argument passed to the recursive function.
`loop` is a Bril program that simply loops `n` times using recursion. `factorial`
is a tail recursive implementation of factorials. `mutual_rec` is a program that
checks whether a program is even or odd in a mutually recursive way. The code for
these can be found [here](https://github.com/sampsyo/bril/pull/37).

**Percentage Change in Memory Usage Using TCE**

|            |   n = 1  |  n = 100 | n = 10000 | n = 100000 |
|:----------:|:--------:|:--------:|:---------:|:----------:|
|    loop    |   +0.1%  |   -2.7%  |   -31.1%  |   -79.5%   |
|  factorial |   +0.2%  |   -2.5%  |   -79.1%  |     X      |
| mutual_rec |   +0.2%  |   +13.8% |   -31.2%  |   -78.4%   |

**Percentage Change in Execution Time Using TCE**

|            |   n = 1  |  n = 100 | n = 10000 | n = 100000 |
|:----------:|:--------:|:--------:|:---------:|:----------:|
|    loop    |  +2.1%   |   +2.2%  |   -10%    |   -29.8%   |
|  factorial |  +1.1%   |   +5.5%  |   -1.6%   |      X     |
| mutual_rec |  +1.1%   |   -1%    |   +5.7%   |   +5.07%   |

To get the execution time and peak memory usage, I use `/usr/bin/time -l` (which prints the contents of rusage).
To make sure the measurements are meaningful, I chose a maximum `n` value so
that the tests took a few seconds. Here we can clearly see that with large values
of `n`, the programs with TCE use considerably less memory. However, it is unclear
whether there is a benefit to the execution time of the program since the values
vary quite a bit.

## Hardest Parts to Get Right
Finding the right level of abstraction for the IR was difficult. I decided to
make it closely resemble x86 because that is familiar and is what matched the
theory the most. The other difficult part was eliminating tail calls that
weren't as simple as just `return foo()`. This required other optimizations and
careful consideration to make sure that indeed the function call could actually
be optimized to just a jump.
