+++
title = "title TBD"



[extra]
bio = """
  Alice is an M.Eng student studying Computer Science. She is broadly interested in compilers, systems and algorithm design.
"""
[[extra.authors]]
name = "Benny Rubin"
[[extra.authors]]
name = "Collin Zhang"
[[extra.authors]]
name = "Alice Sze"
+++

## Performance and Evaluation

The authors compared the performance of SELF with the fastest Smalltalk implementation available and with the standard Sun optimizing C compiler. The Stanford integer benchmarks and the Richards operating system simulation benchmarks were transliterated from C to SELF, SELF' (rewritten in a more SELFish programming style) and Smalltalk. The figure below shows the ratios of the running times of the benchmarks for the given pair of systems.

<!-- <p align="center"> -->
<img src="self-relative-performance.png" width=411 height=300/>
<!-- </p> -->

SELF outperforms Smalltalk on every benchmark by about a factor of two, but is around four to five times slower than an optimizing C compiler. The authors attributed the relative slowness to the quality of the SELF compiler implementation, SELF's robust semantics (e.g. bounds-checking) and the lack of type information. While this is promising for those who want to have their dynamic languages and use them too, some concerns were voiced by the class over the evaluation methods. Firstly, it is unclear how the benchmarks were transliterated, by a human or a program. Either way, it could be that the Smalltalk transliterations were not as good (subjective) as the SELF ones, which gives it an unfair disadvantage. More generally, this highlights the difficulty of using the same benchmarks across different languages. Secondly, real time is used instead of CPU time to measure the running time of Smalltalk, unlike C and SELF, because the two times are "practically identical". But if they are, then why not just use the CPU time for all of them? 