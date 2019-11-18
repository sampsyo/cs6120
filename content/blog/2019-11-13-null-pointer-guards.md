+++
title = "Tail Call Elimination"
[extra]
bio = """
  Christopher Roman is a second semester MEng student in computer science. He is interested in compilers and distributed systems. He is also the best [Melee](https://en.wikipedia.org/wiki/Super_Smash_Bros._Melee) player at Cornell. :)
"""
[[extra.authors]]
name = "Christopher Roman"
+++

## Preface
This blog post is meant for those with limited knowledge of how to use LLVM
to write non-trivial compiler passes. Hopefully this can be a useful resource
to those new to LLVM since, in my experience, information on how to use LLVM
is relatively sparse. I am using LLVM 9. All of the code can be found at
https://github.com/chrisroman/llvm-pass-skeleton.

## Overview
The goal of this project is to add null pointer checks before pointer dereferences
to avoid segfaults. However, it would be wasteful to add a null pointer check to
a pointer dereference when we know for sure the pointer is non-null. For example:
```
int *p = new int(4120);
// No null check necessary here
*p = 6120;
```
Thus, we will do a null pointer analysis that determines, at each instruction,
which pointers may be null. Using this, we can limit the number of extraneous
checks.

To accomplish this, we will create a new compiler pass using LLVM.

## Design
TODO: For the null pointer checks, 

For the null pointer analysis (NPA henceforth), we aim for soundness rather than
completeness. That is, we only mark a pointer as *DefinitelyNonNull* only if we are
guaranteed that the pointer is not null. This way, we are guaranteed that a null
check is removed only if it is really safe to do so.

## Implementation
### Adding the null checks
For null pointer checks, we first created a compiler pass called `AddNullCheckFuncPass`
which adds a new function to a module, which is equivalent to:
```C
void nullcheck(void *p) {
  if (p == nullptr) {
    printf("Found a null pointer. Exiting...\n");
    exit(1);
  }
}
```
Note that this is all done at the IR level. Some difficulties that I encountered
were trying to get the

Then we created a compiler pass called `AddNullCheckPass`. This is a `FunctionPass`
which looks for a `LoadInst` or `StoreInst`, as these are the instructions which
actually dereference a pointer. Let's look at how a dereference gets translated to
LLVM IR:
```C
int *p = ...
*p = 6120
```
is represented in the IR as:
```C
%p = alloca i32*, align 8
...
%1 = load i32*, i32** %p, align 8
store i32 6120, i32* %1, align 4
```

We can see that we're storing `6120` into `%1`, so `%1` is what we need to
perform the null check on. To do so, we can simply call the `nullcheck` function
that was created in the previous `AddNullCheckFuncPass`.

### Implementing the Null Pointer Analysis
Now, we want to elide null checks provided we can determine that the pointer
is non-null at the time of dereferencing. Initially, I figured we could use
the existing alias analyses and check if the pointers alias `nullptr`. However,
when I tried this, it would always show as being `NoAlias`. There is the
`AAResults::pointsToConstantMemory` function, but according to the
[documentation](https://llvm.org/docs/AliasAnalysis.html#the-pointstoconstantmemory-method),
> The pointsToConstantMemory method returns true if and only if the analysis can prove that
> the pointer only points to unchanging memory locations (functions, constant global
> variables, and the null pointer).
This is an issue for two reasons. First, we can't differentiate between pointing
to functions, global variables, or the null pointer. Secondly, this only returns
true when the pointer *only* points to one of these constant locations. This means
that if a pointer may or may not point to the null pointer, the function would
return false. This would be too restrictive for our use case, as we are really
looking for pointers that are *definitely not null* rather than those that are
*definitely null*.

Therfore, to perform this null pointer analysis, we do a dataflow analysis
that is similar to CCP (Conditional Constant Propagation). Our lattice elements
will be a mapping from pointers in the program to `PossiblyNull` or `DefinitelyNonNull`.
(We assume `PossiblyNull` and `DefiniteNonNull` also form a lattice with the former as *Top*
and the latter as *Bottom*).
Our *Top* value will be a map where everything maps to `PossiblyNull`, and *Bottom*
is a map where everything maps to `DefinitelyNonNull`. The analysis is a forward
analysis. The *Meet* operator is simply an elementwise *meet* on elements in the map.
For example, if we have the two lattice elements `X = {p: DefinitelyNonNull, q: PossiblyNull}`
and `Y = {p: DefinitelyNonNull, q: DefinitelyNonNull}`, then `meet(X, Y) = {p: DefinitelyNonNull, q: PossiblyNull}`.

Our transfer function is tricky to get right. Consider
```
store i32* %p, i32** %q, align 8
```
Let's call the the memory that `%q` points to `%deref_q`. Naively, we would just
say that if `%p` was `PossiblyNull`, then `%deref_q` would also be `PossiblyNull`.
While this is necessary, it is not sufficient. Consider the following program:
```C
int *a = new int(6120);
int **p = &a;
*p = nullptr;
*a = 100;
```
Observe that the line `*a = 100` is a null pointer dereference, because
`p` aliases `&a`. That is, the memory location that `p` point to and `&a`
point to are the same. Thus, when we do `*p = nullptr`, we are also setting
`a` to `nullptr`. This means that whenever we a store a value `%p` to `%deref_q`
(the location pointed to by `%q`), we must change everything that aliases `%deref_q`.

One strange thing about this implementation is how to represent `%deref_p`.
Given `store i32* null, i32** %p, align 8`, we need to make sure that subsequent
loads from `%p` are `PossiblyNull`. So, we keep a map `deref_map` from pointers like `%p` to
an arbitrary new pointer. Therefore, when we see `%val = load i32*, i32** %p, align 8`,
we set `lattice[val] = lattice[deref_map[p]]`, where `lattice[x]` tells us if
`x` is `PossiblyNull` or `DefinitelyNonNull`.

## Evaluation
TODO: Do we make a distinction between correctness and performance evaluation?
Depends on what we talk about previously

For correctness,

We also want to see how much of an impact these null checks have on the performance
of programs. Unfortunately I wasn't able to get the PARSEC benchmark tests running.
Instead, I found benchmarks from https://benchmarksgame-team.pages.debian.net/benchmarksgame/
which has various benchmarks for different languages. I chose a small portion of
these tests to run with and without my compiler passes.

I chose to run:
- [binary-trees](https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/binarytrees-gpp-2.html)
  - Ran with: `./binary-trees 13`
- [fannkuch-redux](https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/fannkuchredux-gpp-5.html)
  - Ran with: `./fannkucuh-redux 10`
- [fasta](https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/fasta-gpp-1.html)
  - Ran with: `./fasta 250000`
- [mandelbrot](https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/mandelbrot-gpp-5.html)
  - Ran with: `./mandelbrot 2000`
- [n-body](https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/nbody-gpp-1.html)
  - Ran with: `./nbody 5000000`

Here we simply show the mean time taken to run these programs and the standard
deviation. There are three columns, representing the three different ways we
compiled the program.
- *Baseline* is just compiling using Clang without our null check passes.
Compiled with `/usr/local/opt/llvm/bin/clang++ -O2 <benchmark>.cpp -o <benchmark>`, e.g.,
`/usr/local/opt/llvm/bin/clang++ -O2 binary-trees.cpp -o binary-trees`
- *No NPA* is compiling using the null check passes, but without taking into
consideration which pointers are `PossiblyNull` or `DefinitelyNonNull`. That is,
we do a null check on every pointer dereference.
NOTE: Had to go into source code and remove the check for if pointers are `PossiblyNull`
Compiled with `../tests/compile-with-opt.sh nbody.cpp -O2`
- *With NPA* is compiling using the null check passes *and* eliding null checks
if the pointer is `DefinitelyNonNull`.
Compiled with same instruction as above.

**Runtime Means and Standard Deviations**

|                 |      Baseline      |      No NPA      |     With NPA     |
|:---------------:|:------------------:|:----------------:|:----------------:|
|  binary-trees   |  208 ms ± 4.78 ms  | 217 ms ± 12.8 ms | 213 ms ± 6.78 ms |
|  fannkuch-redux |  222 ms ± 5.02 ms  | 536 ms ± 6.34 ms | 563 ms ± 54.9 ms |
|  fasta          |  200 ms ± 6.01 ms  | 365 ms ± 7.77 ms | 375 ms ± 28.8 ms |
|  mandelbrot     |  319 ms ± 5.79 ms  | 294 ms ± 2.52 ms | 298 ms ± 9.76 ms |
|  n-body         |  327 ms ± 16.7 ms  | 733 ms ± 12.4 ms | 734 ms ± 17.8 ms |

TODO: Discuss the evaluation results a little.

## Hardest Parts to Get Right
1. Talk about the learning curve to LLVM. The documentation is fairly good, but
there's not much information overall on how to do specific things. So a lot of
time was spent just learning how to insert certain nodes into the IR like
function calls.
2. The actual null pointer analysis proved to be more difficult than I thought.
- Can talk about interesting idea where we could have all pointers point to a special

## Extras
- Link to Chandler's talk about optimizations. Talk a little bit about the escape function
that forces variables to be kept around, and how that proved to be useful when debugging
certain functions. 
