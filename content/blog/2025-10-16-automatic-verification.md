+++
title = "From Fragile Hacks to Formal Guarantees: How Alive Made LLVM Safer"

[extra]
latex = false
bio = """
Nikil Shyamsunder, Shihan Fang, Joseph Maheshe, Ruolin Ye, and I-Ting Tsai are all students in Advanced Compilers.
"""

[[extra.authors]]
name = "Nikil Shyamsunder"
[[extra.authors]]
name = "Shihan Fang"
[[extra.authors]]
name = "Joseph Maheshe"
[[extra.authors]]
name = "Ruolin Ye"
[[extra.authors]]
name = "I-Ting Tsai"
+++


## Background: When "Optimizations" Break Your Compiler

Modern compilers apply thousands of peephole optimizations—tiny rewrites that simplify or speed up instruction sequences. For example:

```llvm
%1 = xor i32 %x, -1
%2 = add i32 %1, 3333
```

can be safely replaced by:

```llvm
%2 = sub i32 3332, %x
```

to remove unnecessary operations.  
These optimizations seem trivial, but they’re a minefield. A single missed corner case, say, an overflow or a shift by an invalid amount, can silently break every program compiled with that version. LLVM’s heavy use of undefined behavior makes this even trickier.

The 2015 PLDI paper **"Provably Correct Peephole Optimizations with Alive"** by Lopes et al. introduces a tool designed to make such optimizations *provably safe*. It offers a rare blend of **formal reasoning** and **everyday compiler engineering**.


## The Core Idea: Correctness by Construction

**Alive** is a domain-specific language (DSL) that lets LLVM developers specify transformations declaratively and automatically check whether they are correct using an SMT (Satisfiability Modulo Theories) solver.

Instead of writing and reasoning about C++ code directly, developers describe an optimization at a higher level, and Alive:

- **Verifies correctness.**  
  It checks three logical conditions:  
  (a) the target code is defined whenever the source is;  
  (b) the target doesn’t introduce new "poison" or undefined behavior;  
  (c) both versions produce equivalent results.

- **Models undefined behavior precisely.**  
  Alive captures LLVM’s three nuanced forms—`undef`, `poison`, and true undefined—and enforces correctness even under these relaxed semantics.

- **Infers attributes automatically.**  
  It can infer when to safely add or preserve flags like `nsw` (no signed wrap) or `nuw` (no unsigned wrap), avoiding both over-conservatism and subtle unsoundness.

- **Generates usable LLVM code.**  
  Once proven correct, Alive outputs verified C++ that drops directly into LLVM’s `InstCombine` pass.

Empirically, the authors translated 334 real LLVM optimizations, finding **8 genuine bugs**, most involving unintended undefined behavior. Several of these fixes landed in LLVM itself.


## Practicality: Is Alive Feasible for Everyday Development?

One discussion point that came up repeatedly in class was: *how practical is Alive to use, given that verification can take hours for complex bit-widths or multiplication-heavy rules?*

The authors note that Alive sometimes spends **hours** proving correctness for 64-bit cases, especially involving division or memory. Yet as we discussed, this verification is a **one-time cost**, done at development or CI time—not during every compilation. Spending a few hours once to gain permanent correctness guarantees is, in practice, cheaper than debugging miscompilations months later.

Moreover, much of the runtime variance stems from SMT solver heuristics. Students raised the idea of introducing **lighter-weight heuristics or incomplete checks**, a "fast mode" that provides high confidence but not full proof. Such an approach could make Alive more interactive for day-to-day patch validation while deferring full proofs to overnight CI.

As for **annotation inference**, the SMT-based attribute synthesis is powerful but expensive. Future work could explore simplified constraint systems or symbolic precomputation to make attribute inference more scalable.


## Beyond Peepholes: Toward Global and Control-Flow Optimizations

Alive deliberately focuses on **local transformations**, those that operate within a single SSA expression tree.  
This focus makes verification tractable, but it raises a natural question: can the approach scale to larger, control-flow–based optimizations or interprocedural passes?

Class discussion highlighted the difficulty here: correctness at the local level doesn’t always guarantee correctness globally.  
An optimization might be valid in isolation but alter observable behavior when reordered with memory accesses or across branches. Some LLVM bugs in the past, like incorrect reordering of memory operations, arose exactly this way.

Extending Alive would therefore require reasoning about context: defining equivalence in terms of whole-program semantics rather than isolated snippets. One possible direction would be to design higher-level abstractions over LLVM IR (e.g., summarizing memory and control effects) so that Alive could verify that a local rewrite remains correct in any surrounding program.

This would bridge the gap between local verification and contextual equivalence, ensuring that optimizations proven safe in isolation remain safe when composed in larger pipelines.


## Composition: When Verified Optimizations Interfere

Another recurring question was whether individually correct optimizations can still go wrong when composed. In principle, yes, if one pass breaks a precondition expected by another.

Alive verifies each optimization in isolation, not the interactions between them. While its formalism guarantees that each rewrite preserves semantics on its own, compositional correctness still depends on LLVM’s broader design discipline. If an earlier pass removes attributes that a later one assumes, correctness could fail despite both being "provably correct" individually.

This points to a broader notion of the **Trusted Computing Base (TCB):** we trust that (1) LLVM’s semantics are faithfully modeled, and (2) the C++ code generator from Alive to LLVM is itself correct. Any divergence there, say, if LLVM IR evolves, would lie outside the formal guarantees.

Still, Alive’s focus on LLVM’s well-defined IR gives it an edge. Unlike older compilers whose optimization passes manipulated opaque internal data structures, LLVM’s typed SSA form provides a shared semantics that makes such reasoning possible at all.


## Correctness vs. Performance: A Worthwhile Tradeoff?

One of the liveliest debates centered on whether full formal verification is worth its computational cost.

Alive’s proof generation can be slow, but this slowness only happens once per optimization. Compared to the potential cost of miscompilation, the tradeoff is almost always favorable, especially for infrastructure used by thousands of downstream projects.

Several students noted, though, that compiler designers sometimes accept "slightly wrong but fast" optimizations in less safety-critical domains, a nod to the field of approximate computing. For systems software, however, correctness overwhelmingly wins. As one classmate summarized: "I’d rather spend hours proving one optimization right than weeks tracking down one that silently broke user code."



## Broader Impact and Future Outlook

Alive has since inspired successors like **Alive2**, **Souper**, and verification frameworks for other IRs. Its success illustrates that formal methods don’t have to be all-or-nothing; they can be incremental, usable, and developer-friendly.

By embedding formal reasoning inside everyday engineering workflows, Alive turned proof generation from an academic exercise into a practical safeguard for production compilers.

Looking forward, integrating heuristic verification modes, multi-pass reasoning, and control-flow abstraction could bring Alive’s spirit to the next generation of compiler infrastructures, helping ensure that "optimization" never again means "maybe wrong, but faster."



## References

 Lopes, N. P., Menendez, D., Nagarakatte, S., & Regehr, J. (2015). *Provably Correct Peephole Optimizations with Alive.* PLDI 2015.
