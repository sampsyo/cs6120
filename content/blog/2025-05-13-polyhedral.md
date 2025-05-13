+++
title = "A Simplified Polyhedral IR for Bril"
[[extra.authors]]
name = "Mark Barbone"
+++

In [this project][github], I set out to build a polyhedral IR for Bril
programs.  My original goal was to implement polyhedral optimizations on it;
however, simply constructing and going out of a polyhedral IR ended up being
complex enough that it filled my budget.

[github]: https://github.com/mb64/cs6120-work/tree/main/poly

## Background

Here's an example program:

```c
for (i = 0; i < N; i++) {
    for (j = i; j < N; j++) {
        foo(i, j);
    }
}
```

This program has a pretty complex control flow graph -- each `for` loop is
hiding whole host of basic blocks that it compiles to.  Often, this is great
for a compiler, as breaking up the control flow into small but regular pieces
means that general-purpose optimizations can handle all kinds of programs
uniformly.

However, sometimes we want to do more high-level optimizations. For example, it
may be possible to interchange the two loops:

```c
for (j = 0; j < N; j++) {
    for (i = 0; i <= j; i++) {
        foo(i, j);
    }
}
```

Figuring out how to do this interchange correctly is extremely non-trivial,
even on the high-level `for`-loop-containing program text! For this, it would
be very convenient to have some intermediate language that explicitly accounts
for iteration spaces like `{ [i, j] : 0 <= i and i <= j and j < N }`, so that
these transformations would be possible. This is precisely the goal of the
_polyhedral model_.  In my IR, this is represented as:

```c
for (i ∈ { [i] : 0 <= i < N }) {
    for (j ∈ { [i, j] : 0 <= i <= j < N }) {
        foo(i, j);
    }
}
```

Here, the iteration domain appears as a first-class entity in the IR's looping
constructs. I don't exactly use the classic polyhedral framework, but rather a
simplified form inspired by [MLIR's simplified form][mlir], and I'll explain
the details in the next section.


## The intermediate representation

A classic polyhedral IR has a bag of ordered statements, each with an
associated domain of points on which it's run, and an associated schedule of
when to run it, in the form of an affine function outputting a timestamp.

With this classical representation, turning it all back to code in the end is
very hard: you need to come up with an optimal set of loops and `if` statements
that correctly implements the ordering requested by the schedule, all without
incurring too much blow-up or obscuring hot loops with too many conditionals.
Here are a few papers I found on how to do this, roughly ordered by how complex
their algorithms are: [IJPP'00][ijpp-00], [PACT'04][pact-04], [TOPLAS'15][toplas-15].

[ijpp-00]: https://link.springer.com/article/10.1023/A:1007554627716
[pact-04]: https://icps.u-strasbg.fr/~bastoul/research/papers/Bas04-PACT.pdf
[toplas-15]: https://lirias.kuleuven.be/retrieve/319112

Because of this (and other reasons), [MLIR introduced][mlir] a _simplified_
polyhedral representation, which has an explicit representation of iteration
domains, and the analysis / transformation benefits that affords, but also a
preferred way of iterating over it.

I copied their ideas to design my own IR, which has iteration domains
represented as integer sets (using ISL, the Integer Set Library), but also has
a relatively straightforward interpretation as code:

```
s ::= for (i ∈ affine domain) { ss }        (single-variable for loop)
   |  if (affine condition) { ss }          (conditionals)
   |  instr_1; ...; instr_N                 (basic blocks)

ss ::= s_1; ...; s_N
```

Here, both `affine domain` and `affine condition` are integer sets; the only
difference is that `affine domain` introduces a new variable, while `affine
condition` does not.

A core insight is that we have _static control flow_: there is a subset of
variables, which I'll call "affine variables", which are only transformed in
very easy-to-reason-about ways, and moreover, control flow only depends on
these variables. Of course, not every program has this format, but we still
want to be able to optimize the programs that do, while leaving those that
don't.

## To `affine`-ity, and beyond!

We've made the process of emitting code out of our IR easier, with the
simplified polyhedral representation. But how do we get it there in the first
place? This is still fairly tricky!

Static control flow is a subset of structured control flow, so it makes sense
to start with a program with structured control flow. Bril doesn't have
structured control flow, but we know that it's possible to construct if you
have reducible control flow. So the first step is to reconstruct the structured
control flow of the input Bril program. This is the job of the relooper.

Given a Bril function as input, the relooper either:

* raises an exception, if it has irreducible control flow, or
* returns a Bril AST with structured control flow.

Here's what this structured control flow IR looks like:

```
c ::= block .l { c1 } c2            (l is in scope for c1 but not c2)
   |  loop .l { c }                 (l is in scope for c)
   |  instr_1; ...; instr_N; c
   |  if b then c else c
   |  jmp .l
   |  ret v
```

A jump to label `.l` is only allowed when `.l` is in scope (this is what makes
it structured!). Jumping to a `block` label is like a `break`, escaping to the
end of the block, while jumping to a `loop` label is like a `continue`, going
back to the loop header for the next iteration.

I mostly follow the algorithm from [Beyond Relooper (PLDI'22 functional
pearl)][beyond-relooper], though my structured IR looks a little different, and
I don't make any effort to do anything with irreducible CFGs.

[beyond-relooper]: https://dl.acm.org/doi/abs/10.1145/3547621

This is far more structured than a sea of basic blocks, but still a far cry
from our nice static control flow!  The next step is to recursively process the
structured IR, eventually getting to the simplified polyhedral representation.
This step is by far the most complex, accounting for a large portion of the
code, and with many tricky cases to consider.  The biggest difference between
the structured IR and the polyhedral IR, other than that the polyhedral IR has
static control flow, is that code in the structured IR can have multiple exits:
you can branch to any label that's in scope, depending on whatever conditions
you want. By contrast, the polyhedral IR can only go on to the next statement.
The combination of both multiple exits and recovering static control flow makes
this a complex implementation task.

[mlir]: https://mlir.llvm.org/docs/Rationale/RationaleSimplifiedPolyhedralForm/

## Evaluation

I evaluated the code in three different ways:

 1. Correctness
 2. Performance overhead incurred by round-tripping through the IR
 3. Applicability of potential polyhedral optimizations

For all three, I used both the core Bril benchmark suite and the memory
benchmark suite.  Sadly, as I didn't get to implement any real polyhedral
optimizations, I don't have speed-ups to measure (yet?).

**Correctness.** I tested correctness of round-tripping just through the
relooper, and then also through the polyhedral IR, by comparing against
interpreting the original code.

**Performance.** On the combined mem and core benchmarks, round-tripping
through the polyhedral IR gives a mean 0.78x speedup. However, my compiler at
the moment does zero backend optimizations, which would be quite useful to
clean up the code. I expect this to go back to close to 1x with simple backend
clean-up optimizations.

**Applicability.** Not every program has static control flow, so only some
could potentially benefit from future polyhedral optimizations. In the combined
mem and core benchmarks, 176 out of 272 functions had fully static control
flow, or around 65%. Considering just the relooper, interestingly, all 272
functions in these combined test suites exhibit reducible control flow, and
successfully round-trip through just the structured control flow IR.

