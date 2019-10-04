+++
title = "C Implementation of Bril"
extra.author = "Bhargava S. Manja"
extra.bio = """
  [Bhargava S. Manja](https://github.com/bhargee) is a first year PhD student
interested in computer vision, fast computing, and domain specific languages.
"""
+++

## Introduction

While Bril is a very useful testbed for exploring existing language
technologies and experimenting with new ideas, I was frustrated with the
tooling around the language. It took me three or four hours of fiddling around
with Node, npm, and Python to get `brili` working, and I did not get either
`bril2json` or `bril2txt` to work on my machine (I reimplemented them myself in
Python). I've rarely had such issues with my trusted systems programming
language, C. I decided to implement a simple, fast, and correct interpreter for
Bril. I call it `cril`.

All code can be found in the [project
repository](https://github.com/Bhargee/cril).
  
## Design 

The goal here is simplicity and speed (in comparison to other student
interpreters). The only external library used was for [parsing
JSON](https://github.com/kgabis/parson). The interpreter's evaluation loop, in
`src/interp.c`, first loops through the parsed Bril program and notes the
indices of labels. This was the simplest way to implement jumps and branches,
which become a simple setting of the instruction pointer to the label's index
(or an error if the label is not found in the label->index map). Actual
instructions are implemented with a set of functions, one for each op code.
Once the `op` is known, the interpreter calls one of these functions, which
fetches arguments, does the required manipulation, and stores its result in the
right place (or in the case of effect operations, has the correct effect). 

## Op Code Implementation
All instructions but `jmp`, `br`, `print`, and `const` have a trivial
implementation. A function called `get_or_quit` fetches arguments from the
source program and stores their values in a global array, sized to the max of
the argument count of all non-print instructions (2). If the variable is not
found in storage, an error is reported (containing the specific issue,
incorrect variable, and instruction pointer value) and the program quits.
Otherwise, the op implementing function calls `put` with the argument that
implements the specific op code, which stores the result in the destination
variable. For example, the `add` op code is implemented
as follows:

```c
static void op_add() {
  get_or_quit();
  put(mem_args[0]+mem_args[1]);
}
```
Above, `mem_args` is the name of the global argument array. Thus, most op codes
have 2 line implementations. Even `br`, the most complex op code, requires only
16 lines to implement. 

## Memory Implentation (and More, for Free)
I needed a hash table implementation to implement variable storage, so I built
it myself. It can be found in `src/table.{c,h}`. The dictionary uses open
addressing and linear chaining. The hash function copies the hash used by
[java.lang.String’s hashCode()
method](https://docs.oracle.com/javase/7/docs/api/java/lang/String.html#hashCode()). The `_hash` function computes
a polynomial whose coeffecients are the integer values of the string's
characters, evaluated at `x=31`. The polynomial is evaluated with Horner's
method. The `table` maps string keys to `int64_t` values.  I represent bril
integers and bril booleans with `int64_t` to avoid storing the type of bril
variables. Since bril's typesystem only allows for those two types, this design
is sufficient for now. Incorporating more types will be simple: I can extend
the `table_elem` struct with a type bitfield/enum value.  For now, this simple
table suffices. 

The table code is also used for the label->index map and to store the mapping
between string op codes and the index for that op code's implementing function
in an array of function pointers.

## Evaluation
### Correctness
I ran `cril` against the programs in `bril/test` and made sure the outputs
matched `<program>.out`. My interpreter gave correct results on the Fibonacci
program in `benchmark/fibonacci.json`, while the `brili` reference interpreter
gave wrong results on the last 3 outputted Fibonacci numbers due to rounding
issues with Javascript's `BigInt` type. 

### Performance
#### Benchmark
I collected relatively compute heavy Bril programs under `benchmark` in the
cril repository. I took these programs from Wen-Ding Li's [Bril benchmark
repository](https://github.com/xu3kev/bril-benchmark/tree/master). I used his
Fibonacci and factorial implementation verbatim, and used his matrix
multiplication and polynomial multiplication programs to generate Bril programs
with options `n=1` to `n=5` (the size of the respective matrices and
polynomials). 

#### Measurement and Comparison
I wanted to have a way to get reliable performance numbers to square off
against any other students' implementation of bril. First, I had the main
evaluation loop return a `uint64_t` number of nanoseconds of elapsed time.
I used the POSIX provided `timespec` structure to record the time tracked by
`CLOCK_PROCESS_CPUTIME_ID`, the nanosecond resolution process time clock. This
tracks CPU ticks spent on the program process itself, irrespective of other
scheduled processes. This data is procured with the C standard library's
`clock_gettime`. Some notes: 
* I start timing on after JSON parsing. I did not want that to be
  included. I do the same in my evaluation of the reference interpreter
* I stop timing after cleaning up all data structures used for interpretation

Then, in `src/main.c`, I have a constant named `NUM_RUNS`, and if `cril` is
called with `--benchmark`, it will run each program under `benchmark` `NUM_RUNS`
times, calculate a mean and standard deviation per program, and output the
results. I modified the reference `brili` source code to perform the same
measurements on the same benchmark programs. Times reported are in
milliseconds, displayed as means plus or minus standard deviations. 

| Program | Cril  | Brili |
| :---    | :---  | :---  |
| Fibonacci | .099 ± .03 | .44 ± .23 |
| Factorial | .019 ± .005 | .042 ± .073 |
| MatMul 1  | .005 ± .002 | .004 ± .001 |
| MatMul 2  | .014 ± .004 | .022 ± .025 |
| MatMul 3  | .037 ± .008 | .041 ± .027 |
| MatMul 4  | .08 ± .012  | .075 ± .05  |
| MatMul 5  | .017 ± .023 | .136 ± .07  |
| PolyMul 1 | .005 ± .002 | .006 ± .002 |
| PolyMul 2 | .022 ± .014 | .011 ± .004 |
| PolyMul 3 | .018 ± .003 | .019 ± .005 |
| PolyMul 4 | .028 ± .006 | .027 ± .015 |
| PolyMul 5 | .04 ± .008  | .04 ± .027  |
