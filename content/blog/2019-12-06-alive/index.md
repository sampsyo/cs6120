+++
title = "Provably Correct Peephole Optimizations with Alive"
extra.author = "Alexa VanHattum"
extra.author_link = "https://www.cs.cornell.edu/~avh/"
extra.bio = """
  [Alexa VanHattum](https://www.cs.cornell.edu/~avh/) is a second-year student interested in the intersection of compilers and formal methods. She also enjoys feminist book clubs and cooking elaborate [fish truck][] meals.
"""
+++
In previous discussions, we've considered research systems that find bugs in compiler implementations via _differential testing_.
To page you back in, [CSmith][] and [Equivalence Modulo Inputs (Orion)][emi] both used clever tactics to generate randomized test programs and inputs, with the goal of finding instances where compilers produce different output than expected.
These system exploit a key assumption: while wee don't have an oracle that determines the ground truth correct behavior for any program in the presence of undefined behavior, we can expect compilers to produce the "same" behavior across different implementations.

On the other hand, there are fully verified compilers such as CompCert that guarantee against mis-compilations, but do so at the cost of supporting entire language surfaces and getting fast, optimized code.

What about middle ground, where we leverage a correctness oracle for some particularly tricky portions of a massive, commonly-used optimizing compiler?

Lopes et al.’s [“Provably Correct Peephole Optimizations with Alive”][paper], from PLDI 2015, takes one flavor of this approach.
Instead of treating the compiler itself as a black-box system that we try to break from the outside, Alive _proves_ that the high-level insights behind certain optimizations are correct.
Alive is built for [LLVM][], our friendly massively-optimizing, ahead-of-time, heavily-used beast of a compiler.
Alive aims to hit a design point that is _both_ practical and formal&mdash;the provable guarantees of verified compiler, for one component of a very pragmatic compiler.

[csmith]: https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/bug-finding/
[emi]: https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/equivalence-modulo-inputs/
[paper]: https://dl.acm.org/citation.cfm?id=2737965
[LLVM]: https://llvm.org

### Peephole optimizations

In particular, Alive focuses on LLVM's peephole optimizations&mdash;those that involve replacing a small set of (typically adjacent) instructions with an equivalent, faster set.
For example, a clever compiler might replace `%x = mul i8 %y, 2` (`x = y * 2`) with `%x = shl i8 %y, 1` (`x` = `y` [shift left][shl] `1`).
While these optimizations may ["delight hackers"][delight], they are also extremely tricky to get right for edge cases and boundary conditions.
Alive's specific focus was inspired by the author's previous work on [CSmith][], which found that the single buggiest file in LLVM was`InstCombine` (instruction-combine), the home of over 20,000-C++-lines (!) of peephole optimizations.
Since its publication in 2015, Alive has been used to fix and prevent dozens of bugs and improve code concision in production LLVM.

[shl]: https://en.wikipedia.org/wiki/Arithmetic_shift
[delight]: https://dl.acm.org/citation.cfm?id=2462741

<!-- To see why this specific type of optimization is tricky, let's consider a notoriously annoying boundary condition for programmers&mdash;integer overflow!
Our enterprising compiler engineer might want to exploit their mathematical intuition that for any number `x`, `x + 1 > x`.

 `%x = add %y, %y` (`x = y + y`) with the faster `%x = shl %y, 1`.
While our basic math intuition sense might tell us that this optimization is just as correct as the previous one, this optimization is not allowed in LLVM.
LLVM has a special value, `undef`, that represents _deferred_ undefined behavior.
TODO -->

## System overview

Below is a high-level overview of Alive's approach.

First, Alive comes with it's own domain-specific language (DSL) that was designed to resemble LLVM's intermediate representation.
Optimization are written in this DSL with a source (left hand side) and and target (right hand side) template, which abstract over constant values and exact data types.
The semantics of each side are encoding into logical formulas.
Then, Alive generates verification conditions that cover the full range of potential cases, including special treatment of undefined behavior.
The verification conditions are handed to an off-the-shelf SMT (Satisfiability Modulo Theory) solver, [Z3][], which proves their validity of provides a counterexample.
If the verification conditions are provably correct, Alive is able to generate C++ code that implements the optimization (which the developer can then link into LLVM).
If the verification conditions fail, Alive provides the developer with a counter example in terms of the original source and target template.

[z3]: https://github.com/Z3Prover/z3

<img src="sys-diagram.png" width="700" >

<!-- - Introduce DSL
- Show a simple correct/incorrect optimization
- Diagram of flow -->

## Grokking undefined behavior
- Undefined behavior != unsafe programming!
- Undef vs. poison

## Evaluation/impact
- Ported existing ones: 8 bugs found in InstCombine
- LLVM+Alive performance (discuss poor test coverage)
- “dozens” found in WIP patches (include screenshots)

Ongoing impact
- Incorporated into code review for InstCombine
- Introduction of “freeze” in LLVM IR

## Key take-aways
- DSL + SMT useful for verifying compilers correctness
- Trickiest part, for compiler engineers and verification hackers, is reasoning about UB
- Building robust, usable systems has real world payoff! Benefit of providing small counterexamples
