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
proof-assistant [CITE], or (2) proving that each compiler pass preserves the
observable semantics of a program by checking the equivalence of the input and
the output programs.

Correct by construction compiler have been demonstrated to viable for
non-trivial but require several man-years of work to implement, specify [^1],
and prove correct. On the other hand, proving program equivalence automatically
is a [remarkably hard problem](https://en.wikipedia.org/wiki/Turing_completeness)
which forces such verification efforts to somehow bound the space of program
behaviors.

For our project, we decided to implement the second style of compiler verifier
and prove the correctness of a local value numbering.


[^1]: The problem of specifying the correctness condition of a compiler is itself
a non-trivial, open research problem. Should the compiler preserve the stdout
behavior, or should it give even stronger guarantees such as preserving the
timing behaviour [[CITE]]?
