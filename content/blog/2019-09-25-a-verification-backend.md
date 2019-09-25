+++
title = "Shrimp: Verifying IRs with Rosette"
extra.authors = { "Rachit Nigam" = "https://rachitnigam.com", "Sam Thomas" = "" }
extra.bio = """
  [Rachit Nigam](https://rachitnigam.com) is a second year PhD student interested
  programming languages & computer architecture. In his free time, he
  [subtweets](https://twitter.com/notypes/status/1170037148290080771) his advisor and [annoys tenured professors](https://twitter.com/natefoster/status/1074401015565291520).

  Sam Thomas [[TODO]]
"""
+++


## The Problem

Writing programs is famously hard. Writing program that generate programs
(compilers) is harder still. Compiler verification usually comes in two
flavors: (1) Proving a compiler is correct by construction using a
[proof-assistant][coq], or (2) proving that each compiler pass preserves the
observable semantics of a program by checking the equivalence of the input and
the output programs.

Correct by construction compiler have been demonstrated to viable for
non-trivial but require several man-years of work to implement, specify [^1],
and prove correct. On the other hand, proving program equivalence automatically
is a [remarkably hard problem](https://en.wikipedia.org/wiki/Turing_completeness)
which forces such verification efforts to somehow bound the space of program
behaviors.

For our project, we implemented a pass verification infrastructure for Bril using the
[Rosette][] framework and verified the correctness of a local value numbering
pass.

## SMT Solving, Briefy

Underyling a lot of automatic proof generation is SMT solving. Satisfiability
Modulo Theories (SMT) is a generalization of the [SAT][] problem that allows us
to augment our logic with various "theories" (naturals, rationals, arrays, etc.)
to prove properties in a domain that we care about. For example, SAT + theory of
integers can be used to solve [Integer Linear Programming][ilp] problems.

Program properties can be verified by first encoding the semantics of your
language as an SMT formula and asking a solver to prove it's correctness.

## Rosette

Rosette is a symbolic execution engine for the [Racket][] programming language.
It allows us to write simple Racket programs and automatically lifts them
to perform symbolic computations.

Consider the following program:

    def add(x):
      return x + 1

In addition to running this program with _concrete_ inputs (like `1`), Rosette
allows us to run it with a _symbolic input_. When computing with symbolic
inputs, Rosette _lifts_ operations like `+` to return symbolic formulas
instead.  So, running this program with the symbolic input `x` would give us
the ouput `x + 1`.

Next, Rosette can be used to ask _verification queries_ using a symbolic inputs.
We can write the following program:

    symbolic x integer?
    verify (forall x. add(x) > x)

Rosette will convert this into an SMT formula and verify it's correctness using
a backend solver.

If we give Rosette a falsifiable formula:

    symbolic x integer?
    verify (forall x. add(x) < x)

Rosette generate a _model_ where the formula is false. In this case, Rosette
will report that when `x = 0`, this formula is false.

## Symbolic Interpretation

## Approach

### Basic BLocks

### Downfalls

## Evaulation

To evaluate Shrimp, we performed a case study
where we implemented [Common sub-expression elimination (CSE)][cse] 
using [Local value numbering (LVN)][lvn]
and tested whether Shrimp would find bugs in the implementation. We intentionally planted two bugs 
and found a third bug in the process of testing.

There are some subtleties to a correct implementation of LVN. If you know that the variable 
`sum1` holds the value `a + b`, you have to make sure that `sum1` is not assigned to again before
you use it. For example, consider the following Bril program:
```
sum1: int = add a b;
sum1: int = id a;
sum2: int = add a b;
prod: int = mul sum1 sum2;
```
We would like to replace `sum2: int = add a b` with `sum2: int = id sum1` because we
have already computed the value. However, if we can't do this directly because then `sum2` would
have the value `a`, not `a + b`. The solution is to rename the first instance of `sum1` to something unique so that we don't lose our reference to the value `a + b`. We can
then replace `sum2` with a copy from this new variable.

Shrimp was able to catch this bug and even produce a counter example that proves that the
optimized code produced a different result from the original. With this information,
it is easy to walk through the execution of the code and discover the source of the bug.

The second bug we tested with can come up when extending CSE to deal with associativity.
It would be nice if the compiler knew that `a + b` is equal to `b + a`. The most
naïve thing to do is sort the arguments of values when you compare them so that
`a + b` is the same value as `b + a`. However, this by itself is not enough.
Testing the following example with Shrimp reveals the problem:
```
     sub1: int = sub a b;
     sub2: int = sub b a;
     prod: int = mul sub1 sub2;
```
Shrimp gives us the counter example `a = -8, b = -4`. The problem is that we can't
sort the arguments for every instruction; `a - b != b - a`. Shrimp helps to reveal
this problem.

The final bug was actually an unintentional bug that Shrimp helped us find. We have a
messy internal representation of the Bril ast where each instruction has it's own structure
and is a sub-type of the `dest-instr` structure. When we were looking up values in the LVN table,
we were only comparing that fields in `dest-instr` where the same. This meant that we were
forgetting to actually compare that the op-codes of the instructions where the same!
Shrimp was able to reveal this code from the following example:
```
sub1: int = sub a b;
sub1: int = add a b;
sub2: int = sub b a;
prod: int = mul sub1 sub2;
```
This is the strongest testament that Shrimp is useful in finding bugs in optimization passes.
The moral of the story is that you should use bad code when implementing optimizations
for your bug finding tool so that you can expose real bugs.

## Conclusion
Serval stuff


[^1]: The problem of specifying the correctness condition of a compiler is itself
a non-trivial, open research problem. Should the compiler preserve the stdout
behavior, or should it give even stronger guarantees such as preserving the
timing behavior [[CITE]]?

[rosette]: https://emina.github.io/rosette/
[coq]: https://coq.inria.fr/
[cse]: https://en.wikipedia.org/wiki/Common_subexpression_elimination
[lvn]: https://en.wikipedia.org/wiki/Value_numbering#Local_value_numbering
[sat]:
[ilp]:
