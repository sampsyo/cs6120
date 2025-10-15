+++
title = "Building SSA the Lazy Way: Do We Really Benefit from It?"

[extra]
latex = false
bio = """
Jiale, Nate, Ning, Ziyang, and Amanda are all students in Advanced Compilers.
"""

[[extra.authors]]
name = "Jiale Lao"
[[extra.authors]]
name = "Nate Young"
[[extra.authors]]
name = "Ning Wang"
[[extra.authors]]
name = "Ziyang Chen"
[[extra.authors]]
name = "Amanda Wang"
+++


# Background

Modern compilers almost universally rely on Static Single Assignment (SSA) form, where each variable is assigned exactly once.
SSA makes analyses like constant propagation and dead code elimination simpler and faster, because every use of a variable has a single, unambiguous definition.

**What is Static Single Assignment (SSA).** SSA is an intermediate representation (IR) where each variable is assigned exactly once, and every use refers to a unique definition. This makes data-flow explicit and tends to simplify classic optimizations such as constant propagation, copy propagation, common subexpression elimination, and global value numbering. φ nodes are inserted at control-flow join points to merge values coming from different predecessors.

**Classic SSA construction.** The standard route—see [Cytron et al., 1991](https://dl.acm.org/doi/pdf/10.1145/115372.115320)—inserts φ-nodes (merge operations) at dominance frontiers and relies on precomputed control-flow graphs (CFGs), dominance trees, and sometimes liveness data.
It’s elegant and provably minimal, but it’s also fairly heavy, involving multiple passes over the program before you even get SSA IR.

---

## ⚙️ What Braun et al. (2013) Contribute

> **Build SSA lazily, on demand, as the code is translated.**

Their algorithm:
1. **Constructs SSA directly from the frontend** (AST or bytecode), without first building a non-SSA CFG.  
2. **Finds definitions backward** from each use; only when a variable has multiple incoming definitions do they create a φ-node.  
3. **Uses “sealed blocks”** — placeholders that allow SSA construction even before all control-flow edges are known.  
4. **Performs local optimizations “on the fly”** during construction (constant folding, trivial φ removal, copy propagation).  
5. Produces **pruned SSA** (no dead φ’s) and **minimal SSA** for reducible graphs, with a cleanup pass for irreducible ones.

The algorithm runs in near-linear time, matches the Cytron method’s performance on SPEC benchmarks, and yields **~12% smaller IR** with “on-the-fly” optimizations.

---


---

# Thinking on the Strengths and Shortcomings

## Why This Work is Interesting
The idea flips a long-standing compiler tradition:  
Instead of doing global analysis first and constructing SSA as a batch job,  
Braun’s algorithm treats SSA construction as a **lazy lookup problem**—create φ’s only when truly needed.

That’s conceptually elegant: each use drives the SSA structure, not a precomputed dominance relation.  
It’s also *practical* for **JITs or lightweight compilers**, which benefit from avoiding heavy CFG or dominance computation.

## 🧩 Strengths and Shortcomings

### 👍 Strengths
- **Simplicity in theory and code:** No dominance frontiers, no liveness, no upfront CFG analysis.  
- **Immediate local optimization:** Constant folding and trivial φ elimination happen during IR creation.  
- **Pruned by design:** Every φ corresponds to a real use.  
- **JIT-friendly:** Works well when code is generated incrementally.  
- **Inspired later projects:** Modern IR frameworks (Cranelift, MLIR block arguments, GHC experiments) borrow its on-demand SSA ideas.

### 👎 Limitations
- **Irreducible control flow**: Needs a cleanup pass to reach minimal SSA.  
- **Integration difficulty:** Mature compilers (LLVM, GCC) already depend on CFGs and dominators for other passes, so the benefit disappears there.  
- **Scalability uncertainty:** For complex or deeply nested control flow, recursive backtracking might add overhead—something still untested in modern-scale experiments.  
- **Debuggability and engineering inertia:** LLVM’s mem2reg path and dominance analyses are deeply entrenched and tied to debug metadata and later passes.
---


## 🔬 Broader Discussion and Course Reflections

Several themes emerged in class:

### 🧱 1. Design philosophy
Is it better to **have many optimization passes on the same IR** (as Braun does) or to keep passes **modular and independent** (as LLVM does)?  
The consensus: modularity improves maintainability and debugging, and makes it possible for users to re-order the passes (different orders to order different passes would lead to different performance). But it is hard to decide the best order, and sometimes what is nominally a single IR will actually behave like many, with some passes depending on others having run first.

### ⏱️ 2. When is it worth it?
We agreed Braun’s design shines in **JIT** or **runtime compilation** scenarios, where frontends don’t need complex analyses.  
For large offline compilers, dominance and CFG are already built for other purposes, so Cytron’s approach remains simpler and more consistent.
That is, *not maintaining dominators or dominance frontiers may look elegant, but compilers often need them anyway*—so the “savings” can be misleading if other analyses still require those structures.
Additionally, in a general compiler infrastructure with many frontends, adopting the approach of creating SSA directly from ASTs would require each frontend to implement it separately, which could be a lot more work overall than having each frontend create a non-SSA CFG and then using a common pass to convert it to SSA.

### 🧩 3. Open research question
Surprisingly, no comprehensive follow-up evaluation exists comparing these algorithms on modern benchmarks.  
As professor suggested, this could make a *great course project*—comprehensively evaluating Braun vs. Cytron.

### 🔁 4. Lazy optimization as a broader theme
The “lazy” idea extends far beyond SSA. We see similar strategies in caching, JIT specialization, and even profile-guided optimizations—doing work *only when necessary*.

### 🧮 5. Peformance on complex programs
In path profiling, we can know that many paths are never executed. Using Braun’s lazy idea, we may benefit from not constructing on dead regions. But in some complex programs (e.g., highly nested loops), backward lookups may become costly—another open question for experimentation. So it is hard to know whether it performs well on complex programs and we need comprehensive evaluations.

---

## 🕰️ Historical and Modern Context

**Where it sits historically.**  
Cytron et al. (1991) defined the mainstream, dominance-frontier (DF) approach for placing φ nodes and constructing SSA, which powered decades of production compilers (PDF: <https://dl.acm.org/doi/pdf/10.1145/115372.115320>). Braun et al.’s *Simple and Efficient Construction of Static Single Assignment Form* (2013) offers a complementary lineage: **lazy, on-demand SSA construction** that builds SSA directly during translation, inserts φ only when control-flow joins force ambiguity, and keeps the IR pruned as it’s created (PDF: <https://c9x.me/compile/bib/braun13cc.pdf>). Historically, this paper doesn’t dethrone DF-based SSA; it **formalizes a second, equally principled path**—an SSA-first, use-driven construction style.

**Why it mattered at publication.**  
By 2013, many pipelines still promoted from a non-SSA form to SSA (e.g., alloca→mem2reg), incurring intermediate bloat and delaying simple optimizations. The paper demonstrated we can **skip that pre-SSA detour** with sealed-block discipline and memoized backward lookups, often achieving **pruned (and for reducible CFGs, minimal) SSA** while enabling early constant folding and trivial φ elimination.

**Connections to today’s IR ecosystems.**

- **Within LLVM (MemorySSA).** LLVM’s [MemorySSA](https://llvm.org/docs/MemorySSA.html) builds an SSA-like graph over memory operations (with `MemoryPhi`) and maintains it incrementally (see [`MemorySSAUpdater.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Analysis/MemorySSAUpdater.cpp)). While distinct from variable SSA, its “maintain SSA and merge only when needed” flavor closely echoes Braun et al.’s guidance on simple, low-overhead SSA construction.

- **SPIR-V tooling (Vulkan GPU IR).** The SPIR-V optimizer includes an SSA repair/rewrite stage—see [`source/opt/ssa_rewrite_pass.cpp`](https://github.com/KhronosGroup/SPIRV-Tools/blob/main/source/opt/ssa_rewrite_pass.cpp)—that embodies the same practical philosophy: keep IR near-SSA, fix it as you transform, and minimize live ranges and bookkeeping.

- **MIR JIT toolkit.** The lightweight MIR JIT/IR project adopts Braun et al.’s “simple and efficient” approach for building SSA in a small, fast compiler setting, emphasizing low constant factors and straightforward maintenance. Repo: <https://github.com/vnmakarov/mir>.


# References

- Cytron, R., Ferrante, J., Rosen, B. K., Wegman, M. N., & Zadeck, F. K. (1991). *Efficiently Computing Static Single Assignment Form and the Control Dependence Graph*. ACM TOPLAS. PDF: <https://dl.acm.org/doi/pdf/10.1145/115372.115320>
- Braun, M., Buchwald, S., Hack, S., Leißa, R., Mallon, C., & Zwinkau, A. (2013). *Simple and Efficient Construction of Static Single Assignment Form*. In Proc. CC (ETAPS). PDF: <https://c9x.me/compile/bib/braun13cc.pdf>