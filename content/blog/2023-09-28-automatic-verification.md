+++
title = "Provably Correct Peephole Optimizations with Alive"
[extra]
bio = """
  Benjamin Carleton is a 2nd year PhD student interested in programming languages and formal methods.
"""
[[extra.authors]]
name = "Benjamin Carleton"
+++

Peephole optimizations are a relatively simple yet important class of optimizations performed by compilers. They comprise optimizations that rewrite short sequences of instructions to improve performance. For example, a compiler might replace the instruction `a *= 2` with the semantically equivalent instruction `a <<= 1`, which is almost certainly faster. Such a transformation can be performed by inspecting just the instruction in isolation, without expensive non-local analyses.

The apparent simplicity of these optimizations is somewhat deceptive: correctly reasoning about program equivalence in the presence of undefined behavior is tricky, and bugs are common. This week's paper, [Provably Correct Peephole Optimizations with Alive](https://dl.acm.org/doi/10.1145/2813885.2737965) (Lopes, Menendez, Nagarakatte, and Regehr), presents a domain-specific language for automatically verifying peephole optimizations.

The Alive language and tooling seeks to balance formal methods with usability. The authors contrast this philosophy with that of traditional approaches to correctness: Unlike testing, Alive is sound under mild assumptions. Unlike compiler verification (à la [CompCert](https://dl.acm.org/doi/10.1145/1538788.1538814)), Alive requires no proof engineering on the part of the user. This practical approach has allowed Alive to target [LLVM](https://llvm.org/), a modern, fast-moving compiler infrastructure for which formal verification might otherwise be prohibitively costly.

## The Alive Language

The Alive language was designed to closely resemble the LLVM intermediate representation. A transformation is specified as a source template with an optional precondition, and a target template. For instance, a more general version of the transformation from the introduction might be specified as:

    Pre: isPowerOf2(C1)
    %r = mul %x, C1
      =>
    %r = shl %x, log2(C1)

Templates support integer and pointer operations, but not control flow, and preconditions support various predicates on results from LLVM's dataflow analyses. Additionally, templates may abstract over compile-time constants, bitwidths (note the lack of explicit types in the above example), and various undefined behaviors. The Alive tooling automatically infers feasible types and optimal use of undefined behavior.

### Automatic Verification

Alive's main contribution is a system for automatically checking the correctness of peephole optimizations. The authors accomplish this by encoding transformations into SMT queries, which can then be automatically checked by an off-the-shelf solver such as the [Z3 Theorem Prover](https://github.com/Z3Prover/z3).

Integer operations are naturally encoded into the SMT theory of bitvectors. Alive also supports memory operations, which are encoded using the SMT theory of arrays. In an effort to optimize solver performance, the authors also explore an alternative encoding, termed eager Ackermannization, which yielded faster solver times in their experiments.

More care must be exercised in the encoding of transformations involving undefined behavior. In the presence of undefined behavior, the authors present the following soundness criterion. A transformation is correct when, for each valid type assignment, each of the following hold:

1. The target is defined when the source is defined.
2. The target only produces poison values[^1] when the source does.
3. The target produces the same result as the source when both are defined and the precondition holds.

In the presence of memory operations, a fourth constraint is required:

4. The final memory configuration of the target is equal to that of the source (assuming certain natural constraints on the behavior of the `alloca` instruction).

The result of this encoding is a set of SMT queries that can be used to verify the correctness of a transformation, or else generate a counterexample. The authors bias the solver in an attempt to produce more useful counterexamples, based on the observation that especially large or especially small counterexamples are difficult to understand.

Queries are valid only for a single type assignment, and so correctness must be checked for each valid assignment. To ensure termination, integers are bounded at 64 bits. Though this in theory breaks soundness, the authors note that integers wider than 64 bits are relatively rare in LLVM code.

### Attribute Inference

LLVM IR includes various instruction attributes that introduce undefined behavior, e.g., the `nsw` attribute which makes signed integer overflow undefined. The authors note that such attributes are particularly tricky for developers to use correctly. To address this, they present an algorithm for inferring the optimal placement of attributes. Such a placement minimizes the number of attributes required in the source and maximizes the number of attributes that are propagated to the target; this corresponds to the weakest precondition (so that the optimization is applicable in greatest number of contexts) and the strongest postcondition (to enable the widest range of subsequent optimizations).

### Translation to C++

Transformations specified in the Alive language can be automatically translated to C++. The generated code uses the same pattern matching library as LLVM's hand-written peephole optimizations, enabling its inclusion as part of an LLVM optimization pass.

## Evaluation

The authors translated 334 transformations from LLVM's instruction combiner pass into Alive. Using Alive's automatic verifier, they discovered bugs in eight transformations, all of which were confirmed and fixed by the LLVM developers. Additionally, Alive produced a weaker precondition for one transformation and a stronger postcondition for 70 transformations. In the authors' experience, Alive verification usually takes just a few seconds, but verification of some transformations can take upwards of several hours.

In addition to the authors' testing of existing optimizations, some LLVM developers have adopted Alive to prevent the introduction of new bugs. At the time of publication, Alive had already prevented at least one faulty optimization from being added. Use of Alive's successor, [Alive2](https://dl.acm.org/doi/10.1145/3453483.3454030), continues to the time of this writing.

To test the C++ translator, the authors compiled LLVM with the instruction combiner pass replaced with the Alive-generated transformations; the resulting compiler passed the LLVM test suite. Comparing the Alive-based compiler to vanilla LLVM is not particularly enlightening due to the authors having translated only about a third of the optimizations. Predictably, the Alive version exhibits faster compilation times but slower execution times for the generated code.

## Discussion

The sentiment towards the paper during the class discussion was largely enthusiastic: it is exciting to see work on formal verification that has received adoption in real-world code. In contrast to many other efforts in formal verification, Alive's integration with LLVM requires that it deal with the unmitigated complexity of a real, production compiler, and it does so to great effect. It's encouraging that Alive has continued to see use by LLVM developers, as well as further iteration in subsequent papers.

The authors seem to have found a perfect niche in which formal methods are highly practical. Peephole optimizations are small enough that the proof obligation may be tractably discharged via an off-the-shelf SMT solver, while still being a demonstrably difficult task for human developers. This is especially true in the presence of undefined behavior, and in this respect the authors have empirically demonstrated that their tool can produce better optimizations than those written by hand.

There was also discussion on the size of the trusted computing base, including the relative merits of projects such as [AliveInLean](https://link.springer.com/chapter/10.1007/978-3-030-25543-5_25). Opinions here were more varied, which is natural given the many factors that might influence one's assessment of risk. While bugs in Alive's encodings, its formalization of LLVM's semantics, or its C++ translator are certainly possible, it is a matter of debate whether the considerable effort involved in formally verifying any of these components would be more profitably spent elsewhere.

Finally, many expressed interest in automatically synthesizing peephole optimizations rather than merely verifying them. The paper, along with later work on [Alive-Infer](https://dl.acm.org/doi/abs/10.1145/3062341.3062372), provides ample evidence that developers struggle with finding the strongest possible optimizations. It would be exciting to see an optimization synthesizer that matches Alive's degree of support for real-world IRs.

[^1]: Poison values are a type of deferred undefined behavior: given an illegal operand, an instruction may produce a poison value that only invokes undefined behavior if and when the result is used by an instruction with side effects.
