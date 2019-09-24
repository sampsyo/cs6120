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

For our project, a pass verification infrastructure for Bril using the
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

[^1]: The problem of specifying the correctness condition of a compiler is itself
a non-trivial, open research problem. Should the compiler preserve the stdout
behavior, or should it give even stronger guarantees such as preserving the
timing behavior [[CITE]]?

[rosette]: https://emina.github.io/rosette/
[coq]: https://coq.inria.fr/
[sat]:
[ilp]:
