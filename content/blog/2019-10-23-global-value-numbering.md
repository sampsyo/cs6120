+++
title = "Global Value Numbering"
[extra]
bio = """
  [Alexa VanHattum][] is a second-year student interested in the intersection of compilers and formal methods. She also enjoys feminist book clubs and cooking elaborate [fish truck][] meals.

  [Gregory Yauney][] is a second-year student working on machine learning and digital humanities.
  
[alexa vanhattum]: https://www.cs.cornell.edu/~avh
[gregory yauney]: https://www.cs.cornell.edu/~gyauney
[fish truck]: https://www.triphammermarketplace.com/events/
"""
[[extra.authors]]
name = "Alexa VanHattum"
link = "https://www.cs.cornell.edu/~avh"
[[extra.authors]]
name = "Gregory Yauney"
link = "https://www.cs.cornell.edu/~gyauney"
+++

### Overview

Q. Why GVN?

```
void main(x : int, y : int) {
entry:
  z1 : int = add x y;
  jmp block;
block:
  z2 : int = add x y;
  ret;
}
```

## SSA

Difficulties:
- Dominator tree: we needed the direct children (immediate dominators), not the entire dominance relation. We ran the transitive reduction on the dominance relation.
- We only add a phi node for a variable if that variable is defined more than once.
- Whether to start with phi nodes with duplicated arguments or to add them as we processed the node. We keep track of where the phi argument sources are from.
- Design decision: we also store source blocks for phi arguments.
- Found in `test/gvn/constant-folding-tau-example.bril`: Our implementation requires us to declare variables at a higher scope.

## Global value numbering

### Copy propagation

### Constant folding

Difficulties:
- Recursively visiting blocks in the dominator tree in reverse post order is not enough to ensure that a phi node's arguments have been processed in the absence of backedges. We also had to sort the immedately dominated blocks by their order in the CFG.
- Can't do constant propagation because Bril operations require registers as operands.
- `examples/gvn/constant-folding-calpoly-example.bril`: There's a division by zero when running constant folding, but the compiler shouldn't crash! We filed an issue with Bril's local value numbering and fixed our implementation.


## Evaluation


1. Evaluating correctness: LLVM GVN tests that use only the features that Bril has.

Our implementation does not produce the same output as LLVM on all tests because 1) Bril operations require that all operands are registers and 2) LLVM GVN has more features.

Discuss `test/gvn/briggs-et-al-fig-5.bril`.

2. Trying to get fewer instructions.

| Test program | Original | SSA form | After only LVN | After only GVN |
| :---        |    :----:   | :----: |  :----: |
| `test/gvn/across-basic-blocks.bril` | 8 | 8 | 8 | 7 |
| `test/gvn/copy-propagation.bril` | 5 | 2 | 2 | 2 |
| `test/gvn/constant-folding-tau-example.bril` | 26 | 37 | 37 | 35 |
| `test/gvn/constant-folding-calpoly-example.bril` | 27 | 25 | 25 | 22 |
| `test/gvn/add-commutativity.bril` | 6 | 6 | 6 | 5 |
| `test/gvn/constant_propagation.bril` | 5 | 2 | 2 | 2 |
| `test/gvn/cyclic-phi-handling.bril` | 16 | 22 | 22 | 22 |
| `test/gvn/divide-by-zero.bril` | 5 | 6 | 6 | 6 |
| `test/gvn/equivalent-phis.bril` | 27 | 25 | 25 | 25 |
| `test/gvn/redundant-store-across-block.bril` | 10 | 8 | 8 | 8 |


