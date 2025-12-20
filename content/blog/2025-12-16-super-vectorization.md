+++
title = "Predicated SSA and SLP vectorization"
[extra]
bio = """
  Thomas is a undergraduate in the early M.Eng. program at Cornell University studying Computer Science
"""
[[extra.authors]]
name = "Thomas McFarland"
+++

## Background

### Classical Vectorization

Vectorization is the classic example of SIMD parallelism. Without any confusing parallel primitives, programmers are able to achieve speedups. However, in classical vectorization programs must be structured in specific formats: control flow must be very simple, generally the expectation is that the operation is the only in its control block, and the operation must be isolated from other potentially confounding operations.

Consider a simple loop that can be vectorized:

```c
for (int i = 0; i < n; i ++) {
    c[i] = a[i] + b[i];
}
```

becomes:

```c
for (int i = 0; i < n; i += 4) {
    // Vector add: process 4 elements at once
    vec_a = load(a + i);
    vec_b = load(b + i);
    vec_c = vec_a + vec_b;
    store(c + i, vec_c);
}
```

The newly vectorized loop executes approximately four times faster! However, this vectorization is limited: it only works because the code has a simple loop control flow and the instruction(s) in the loop are straightforward to vectorize.

### Superword Level Parallelism (SLP)

[Superword level parallelism (SLP)](https://dl.acm.org/doi/10.1145/358438.349320), offers a solution to the latter: via grouping of instructions in straight line code by their operation, different instructions can be interleaved such that entire basic blocks can be vectorized. Indeed, with loop unrolling this can achieve the same level of parallelism as classic loop parallelism. In theory there is no limit to this parallelism, but practically SLP rarely expands beyond a single basic block. This arises becauase of the complexity from reordering in a CFG. Thus, optimizations in nearby code is severly limited, even if this would achieve large boosts to speed.

Example of SLP grouping:

```pseudocode
// Original straight-line code
x1 = a1 + b1
x2 = a2 + b2
y1 = c1 * d1
y2 = c2 * d2

// Grouped for vectorization
vec_x = vec_add(vec_a, vec_b)
vec_y = vec_mul(vec_c, vec_d)
```

## The Paper and my Goal

### The Paper's Approach

In ["All You Need Is Superword-Level Parallelism,"](https://dspace.mit.edu/bitstream/handle/1721.1/146343/3519939.3523701.pdf?isAllowed=y&sequence=1) Chen et. al. outline a simple principle: if the compiler can reorder instructions freely and easily determine which instructions can be packed together, the same SLP algorithm which only works in basic blocks suddenly can generalize across a whole function (or, with inlining, a who program). But, LLVM's native CFG based IR does not allow this: moving instructions is inherently tricky, and can easily result in the whole program becoming invalid.

The authors solve this by introducing **Predicated SSA**, which when combined with an SLP algorithm they call **Super Vectorization**. Predicated SSA solves the issue of a CFG via removing the CFG. Instead, a function is one list of straight line code, composed entirely of loops and instructions. To ensure data and control dependencies remain, each instruction or loop has an associated predicate: the code only runs if the predicate is satisfied. Thus, while data dependencies still have to be respected, control dependencies are made irrelevant, turned into a simply equality check

**Traditional CFG:**
```
Start -> If (cond) -> (instr1) -> End
      -> Else -> (instr2) -> End
```

**Predicated SSA:**
```
instr1 (cond)  // runs if cond
instr2 (!cond) // runs if !cond
```

### My Goal

My goal was to implement some form of Super Vectorization, however upon further inspection the paper, while at a fundamental level about vectorization, was more about the form of the IR the paper presented. The authors took almost eight-thousand lines to implement the full pipeline, therefore I limited myself to implementing the IR and pieces of the optimizations. For the IR, a full implementation was the set goal, letting any program in the LLVM IR be translated into Predicated SSA. On optimizations, this was kept fairly minimal, as the point of the project was more the framework than the optimizations themselves. Basic vector packing was achieved, following data dependencies and straight line code.

## Implementation

My implementation is available as [an open-source project on GitHub](https://github.com/tf-mac/superVectorization). I built this in LLVM, meaning a large chunk of the needed code was translation between the LLVM IR and the Supervectorization IR. The overall implemention took about a thousand lines of C++ code. Most of that was spent on the conversion to and from Predicated SSA, as the intention of the IR is to make any post-translation analysis relatively simple. My process for implementing the IR was to outline the semantics, then build up skeletons of both functions and finally fill in the relevant helper functions. The IR itself was a close implementation of the semantics described in the paper: the output of translation is a function, which is a name and a list of "items," which were either an LLVM instruction (or more precisely, pointers to LLVM instructions in the original CFG) of a loop associated with a predicated. Loops, in this IR, are a set of items themselves with an associated loop condition and various "Mu-nodes" which allow for recursive and incrementing behaviour. Predicates were either some function of other predicates (not, or, and), true, or an LLVM value.

### Translation Back to LLVM IR

On the translation back to the LLVM IR, the complexity primarily manifested in the nature of the process itself. First, the process necessitated completely rebuilding the CFG, we couldn't just move around instructions without a heavy cost due to the changes in control flow. In essence, this process required cloning the instructions present in the original CFG, and building up new basic blocks starting from a new entry block. For normal instructions this was fairly trivial to accomplish (chaining basic blocks for "and" and "or" predicates), however loops involved some added complexity. Precisely implementation details are outside the scope of this blog but can be found both in the original paper and in my source code. In addition, the way I structured the IR was primarily as pointers to instructions, meaning that the old CFG had to remain until the new CFG was built. Doing this in LLVM requires a bit of hacking, and the easiest way I found to do it was to access the entry basic block and place a new basic block right before it. Then, I'd insert the new CFG as descendants of that block, and when the process was completed I'd delete the old entry block and its descendants. The second problem arose from the old instructions and values being deleted at the end of the process. Solving this turned out to be fairly trivial; LLVM has an inbuilt tool to map old values to new ones and change instruction mappings accordingly.

### Translation Into Predicated SSA

I found the translation into the IR to be far more difficult and frustrating. The process initially appeared to be straightforward, to the point that the original paper only gave a two paragraph overview of the process. However, it rapidly became clear this was not as straightforward as initially seemed. The large initial problem was my unfamiliarity with LLVM, and it took a long time to get up to speed and successfully design a function that could traverse the instructions and loops of the program and process them accordingly. While this took time, progress was clear. The larger problem was the control predicates. To begin with, these were already difficult at the outset: most control predicates involve a disjunction on a relationship involving the Post-Dominator frontier and successor post-dominator of a basic block. From my research it appears LLVM has removed their internal Post-Dominator frontier tooling, so I had to build that up myself (using the internal tree tools of course). The formulas were also very opaque, and it took a long time to understand the intention. The most frustrating version of this was the realization that the formulas displayed for the control predicates had their branches incorrectly ordered and indeed had some missing cases, to the point that somewhat trivial counter examples can be conjured[^1]. All the same, upon realizing this I implemented a slightly modified version that fixed the most glaring issues. I found this to be the most difficult part of the project, as distinguishing between my own misunderstanding and an error in the paper made the programming much more difficult.

[^1]: The exact details here relate to SSA form involve stores and loads, but an in-depth explanation is outside of the scope here.

### SLP Implementation

The SLP was relatively easy as I said above, I simply had to comb through the instructions, find ones that had the same operation and predicate, and do some rudimentary checking of data dependency. Then, I moved them into packs to allow LLVM to turn these into vector operations on codegen.

## Success

As the character of the project turned from the optimization focus to the IR focus, my metrics for success changed. Initially I planned to measure success in speedups. Rather, I measured it in successful translation to and from the new IR. I hand wrote a variety of C programs and ensured they all still worked after being passed through my machinery, thus while not ensuring correctness giving a degree of certainty (barring the bug mentioned above in the paper on control predicates).[^2] Thus the rigorous proof I have here is that a variety of programs were translated into the new IR and back and returned the same return code as expected. I also wrote a few small programs that had non-trivial control flow optimization, and while the super-vectorizer did not find all of them it was able to group a good number together, which I think is sufficient for a the work I put in. I am most of all proud of the end result, as the 1000 lines of C++, along with the various other outputs produced, show that a large amount of work went into this project. I think, given how large of a lab worked on the original project and how long it took them to compile it, I am quite proud of being able to make something even close to a replica.

[^2]: These were hand written as to test the intermediary step of producing a correct predicated SSA. While a wide range of generic tests could be used as to ensure the output C++ program from this process is equivalent, it would have been difficult to both understand the programs and verify the produced SSA if I had not written them myself. Were I to continue this project I would like to expand the test set to include a wider variety of programs, including blind testing speedups from the vectorization.

#### GAI Statement

I utilized [ChatGPT](https://chatgpt.com/)[^3] during this assignment to understand the LLVM syntax, for example understanding how to use the replace value helper function. Funnily enough, whenever I would try to have it debug my code it would get very confused and start inventing new structs for no reason, so it was surprisingly unhelpful. I still think its a useful tool, especially for something so opaquely documented as LLVM, but it is very flawed.

[^3]: For clarity I used GPT-4.1 within the copilot interface.