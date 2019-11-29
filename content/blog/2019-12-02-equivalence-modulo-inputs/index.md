+++
title = "Compiler Validation via Equivalence Modulo Inputs"
[extra]
bio = """
  [Rolph Recto](rolph-recto.github.io) is a third-year graduate student studying
  the intersection of programming languages, security, and distributed systems.
  [Gregory Yauney](https://www.cs.cornell.edu/~gyauney)
  is a second-year student working on machine learning and
  digital humanities.
"""
latex = true
[[extra.authors]]
name = "Rolph Recto"
link = "rolph-recto.github.io"
[[extra.authors]]
name = "Gregory Yauney"
link = "https://www.cs.cornell.edu/~gyauney"
+++


Imagine you, being a clever person who wants her C programs to run faster,
sat down, thought very hard, and developed a new compiler optimization.
Say you implement it as a transformation pass in LLVM so that other people
can take advantage of your cleverness to make their programs run faster as well.
You run your optimization pass over a few benchmarks and see that it does
indeed make some programs run faster.
But a question nags you: how do you know that your optimization is correct?
That is, how do you know that your optimization doesn't change the semantics
of the input program?

*Equivalence Modulo Inputs*, a testing technique introduced by Le et al
in a [PLDI 2014 paper][paper], allows our compiler hacker above to test
her optimization rigorously without much effort.

* allows debugging optimizations

[paper]: https://dl.acm.org/citation.cfm?id=2594334


## Some definitions


> **EMI(I)-validity**. Given an input set *I*, a compiler *C* is *EMI(I)-valid*
  if for any program *P* and EMI(I)-variant *Q*, it is the case that
  `C(P)(i) = C(Q)(i)` for all *i* in input set I. 
  **If a compiler is not EMI(I)-valid, then we consider it buggy.** 

But the inverse is not true: if a compiler *is* EMI-valid, it can
still be buggy!
Consider the degenerate compiler that maps all source programs to the same
target program.
The compiler is EMI(I)-valid for any input set *I*, but it is obviously buggy.

Thus EMI-validity is a conservative overapproximation for compiler correctness,
which is still useful for finding bugs in practice.
Its failure to find bugs cannot verify a compiler implementation
(read: absence of evidence is not evidence of absence),
but it can give higher assurance that it works as intended.


## EMI in Practice: Orion

How do we


## Evaluation
