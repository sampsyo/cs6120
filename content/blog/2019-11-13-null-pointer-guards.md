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
completeness. That is, we only mark a pointer as *DefinitelyNull* only if we are
guaranteed that the pointer is null. This way, we are guaranteed that a null
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
is non-null at the time of dereferencing. To do so, we do a dataflow analysis
that is similar to CCP (Conditional Constant Propagation). Our lattice elements
will be a mapping from pointers in the program to `PossiblyNull` or `DefinitelyNonNull`.
(We assume `PossiblyNull` and `DefiniteNonNull` also form a lattice with the former as *Top*
and the latter as *Bottom*).
Our *Top* value will be a map where everything maps to `PossiblyNull`, and *Bottom*
is a map where everything maps to `DefinitelyNonNull`. The analysis is a forward
analysis. The *Meet* operator is simply an elementwise *meet* on elements in the map.
For example, if we have the two lattice elements `X = {p: DefinitelyNonNull, q: PossiblyNull}`
and `Y = {p: DefinitelyNonNull, q: DefinitelyNonNull}`, then `meet(X, Y) = {p: DefinitelyNonNull, q: PossiblyNull}`.

Our transfer function is tricky to get right.

## Evaluation
TODO: Do we make a distinction between correctness and performance evaluation?
Depends on what we talk about previously

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
