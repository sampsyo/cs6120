+++
title = "Predicated SSA and SLP vectorization"
[[extra]]
bio = """
  Thomas is a undergraduate in the early M.Eng. program
"""
[[extra.authors]]
name = "Thomas McFarland"
+++

## Background

Vectorization is the classic example of SIMD parallelism. Without any confusing parallel primitives, programmers are able to achieve speedups. However, in classical vectorization programs must be structured in specific formats: control flow must be very simple, generally the expectation is that the operation is the only in its control block, and the operation must be isolated from other potentially confounding operations. Superword level parallelism (SLP), offers a solution to the latter: via grouping of instructions in straight line code by their operation, different instructions can be interleaved such that entire basic blocks can be vectorized. Indeed, with loop unrolling this can achieve the same level of parallelism as classic loop parallelism. In theory there is no limit to this parallelism, but practically SLP rarely expands beyond a single basic block. This arises becauase of the complexity from reordering in a CFG. Thus, optimizations in nearby code is severly limited, even if this would achieve large boosts to speed.

## The Paper and my Goal

In ["All You Need Is Superword-Level Parallelism,"](https://dspace.mit.edu/bitstream/handle/1721.1/146343/3519939.3523701.pdf?isAllowed=y&sequence=1) Chen et. al. outline a simple principle: if the compiler can reorder instructions freely and easily determine which instructions can be packed together, the same SLP algorithm which only works in basic blocks suddenly can generalize across a whole function (or, with inlining, a who program). But, LLVM's native CFG based IR does not allow this: moving instructions is inherently tricky, and can easily result in the whole program becoming invalid.

The authors solve this by introducing Predicated SSA, which when combined with an SLP algorithm they call Super Vectorization. Predicated SSA solves the issue of a CFG via removing the CFG. Instead, a function is one list of straight line code, composed entirely of loops and instructions. To ensure data and control dependencies remain, each instruction or loop has an associated predicate: the code only runs if the predicate is satisfied. Thus, while data dependencies still have to be respect, control dependencies are made irrelevant, turned into a simply equality check

My goal was to implement some form of Super Vectorization, however upon further inspection the paper, while at a fundamental level about vectorization, was more about the form of the IR the paper presented. The authors took almost eight-thousand lines to implement the full pipeline, therefore I limited myself to implementing the IR and pieces of the optimizations. For the IR, a full implementation was the set goal, letting any program in the LLVM IR be translated into Predicated SSA. On optimizations, this was kept fairly minimal, as the point of the project was more the framework than the optimizations themselves. Basic vector packing was achieved, following data dependencies and straight line code.

## Implementation

Source Code: [HERE](https://github.com/tf-mac/superVectorization)

The overall implemention took about a thousand lines of C++ code. Most of that was spent on the conversion to and from Predicated SSA, as the intention of the IR is to make any post-translation analysis relatively simple. My process for implementing the IR was to outline the semantics, then build up skeletons of both functions and finally fill in the relevant helper functions. Both functions had some difficulty both from learning how to implement the ideas and from the paper itself.

On the translation back to the LLVM IR, the complexity primarily manifested in the nature of the process itself. First, the process necessitated completely rebuilding the CFG, we couldn't just move around instructions without a heavy cost due to the changes in control flow. In addition, the way I structured the IR was primarily as pointers to instructions, meaning that the old CFG had to remain until the new CFG was built. Doing this in LLVM requires a bit of hacking, and the easiest way I found to do it was to access the entry basic block and place a new basic block right before it. Then, I'd insert the new CFG as descendants of that block, and when the process was completed I'd delete the old entry block and its descendants. The second problem arose from the old instructions and values being deleted at the end of the process. Solving this turned out to be fairly trivial; LLVM has an inbuilt tool to map old values to new ones and change instruction mappings accordingly.

I found the translation into the IR to be far more difficult and frustrating. The process initially appeared to be straightforward, to the point that the original paper only gave a two paragraph overview of the process. However, it rapidly became clear this was not as straightforward as initially seemed. The large initial problem was my unfamiliarity with LLVM, and it took a long time to get up to speed and successfully design a function that could traverse the instructions and loops of the program and process them accordingly. While this took time, progress was clear. The larger problem was the control predicates. To begin with, these were already difficult at the outset: most control predicates involve a disjunction on a relationship involving the Post-Dominator frontier and successor post-dominator of a basic block. From my research it appears LLVM has removed their internal Post-Dominator frontier tooling, so I had to build that up myself (using the internal tree tools of course). The formulas were also very opaque, and it took a long time to understand the intention. The most frustrating version of this was the realization that the formulas displayed for the control predicates had their branches incorrectly ordered and indeed had some missing cases, to the point that somewhat trivial counter examples can be conjured. All the same, upon realizing this I implemented a slightly modified version that fixed the most glaring issues. I found this to be the most difficult part of the project, as distinguishing between my own misunderstanding and an error in the paper made the programming much more difficult.

## Success


