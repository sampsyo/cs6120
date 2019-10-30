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

## Static Single Assignment (SSA) Form

The primary difference between how you go about actually implementing _local_ value numbering vs. _global_ value numbering is the form of the input source code.
For local value numbering, the analysis can take in code in any standard imperative assignment format.
Because local value numbering only considers one basic block at a time, the compiler can easily determine the relationship between assignments to the same variable (that is, if the source writes to a variable `x` twice, we can just process the code in the block in order to know which assignment is relevant). 
This clean assumption breaks when considering multiple basic blocks.
If `x` is assigned to in two different predecessors of the block we are currently processing, looking up the value number for any assignment involving `x` on the right hand side becomes impossible!

Global value numbering sidesteps this difficulty by requiring that the input source code first be transformed to [_static single assignment (SSA)_][ssa] form. 
In SSA, every variable name can only be assigned to once. 
Reassignments to the same variable in the original code are translated to assignments to new variable names. 
Because reassignments often take place in different control flow branches (to actually have the branches be useful!), SSA form needs a way to recombine different names back into a final variable. 
SSA form relies on a special additional instruction, canonically called `phi` nodes (evocative of `if`, backward) to combine variables from diverging control flow back into a single variable. `phi` instructions take as arguments a list of renamed variables, and assign one of the variables into a new assignment based on which control flow block was actually taken.

For example, consider the following Bril code:

```
start:
  br cond left right;
left:
  x : int = const 1;
  jmp exit;
right:
  x : int = const 2;
  jmp exit;
exit:
  print x;
```

To convert these basic blocks to Bril code, we rename variables and insert a `phi` instruction when the control flow merges again in the last block.

```
start:
  br cond1 left right;
left:
  x1 : int = const 1;
  jmp exit;
right:
  x2 : int = const 2;
  jmp exit;
exit:
  x3 : int = phi x1 x2;
  print x3;
```

Having unique variable assignments is enormously helpful in implementing optimization---most industrial-strength compilers, including [LLVM][] and [GCC][], rely on it internally. We followed a standard algorithm to convert Bril programs to SSA form. 

The first pass inserts `phi` nodes wherever control flow merges multiple variable definitions. 
We determine this condition by computing the [_dominance frontier_][domf] (essentially, boundaries in the control flow graph where control flow can recombine) for every basic block. 
For every block in the dominance frontier of every variable, we insert a `phi` node with as many arguments as there are predecessors to the block. 
One bug we hit with this part of the implementation was realizing that we only should insert `phi` nodes at the dominance frontier for variables that we assigned to multiple times. 
For convenience of later analysis, we also augmented the base algorithm to track which basic blocks (or source) each `phi` argument originated from (for the above example, our generated `phi` would be `x3 : int = phi x1 x2 left right;`)  

The second pass is a recursive renaming algorithm that uses a stack to determine the correct version of a variable name for every point in the program.
The trickiest part of implementing this algorithm was traversing the basic blocks in the correct order.
The high level algorithm specifies that you should recursively traverse the children of each block in the dominator tree. 
However, we found that not only did we ned to traverse the children, but we needed to traverse them _in order_ relative to the control flow graph. 
In particular, Bril’s ecosystem already had a utility for getting the dominators of every block. To get the dominance tree, we calculated the transitive reduction of this relation. 
Our implementation additionally extending the dominance tree such that every list of children was ordered relative to a reverse post-order traversal of the control flow graph.

In later testing (using an example, `test/gvn/constant-folding-tau-example.bril`, ported from the [class notes of Prof. Mooly Sagiv of Tel Aviv University)][tau], we found one minor bug with our SSA implementation. 
Bril programs do not have a type checker and thus are fairly aggressively dynamic: it’s perfectly valid to assign to a variable in one branch of a conditional but not the other. 
The base SSA algorithm does not account for this, thus we fail to properly rename in the second step of our algorithm. 
For now, we work around this case by assigning a default value to to the variable before we branch. 

[ssa]: https://en.wikipedia.org/wiki/Static_single_assignment_form
[llvm]: https://llvm.org/docs/tutorial/OCamlLangImpl7.html
[gcc]: https://gcc.gnu.org/onlinedocs/gccint/SSA.html
[domf]: https://en.wikipedia.org/wiki/Dominator_(graph_theory)
[tau]: http://www.cs.tau.ac.il/~msagiv/courses/pa07/lecture2-notes-update.pdf

## Global value numbering

### Copy propagation

### Constant folding

Difficulties:
- Recursively visiting blocks in the dominator tree in reverse post order is not enough to ensure that a phi node's arguments have been processed in the absence of backedges. 
We also had to sort the immedately dominated blocks by their order in the CFG.
- Can't do constant propagation because Bril operations require registers as operands.
- `examples/gvn/constant-folding-calpoly-example.bril`: There's a division by zero when running constant folding, but the compiler shouldn't crash!
We filed an issue with Bril's local value numbering and fixed our implementation.


## Evaluation


1. Correctness

### Briggs, Cooper, and Simpson

Our implementation does not produce the same output as LLVM on all tests because 1) Bril operations require that all operands are registers and 2) LLVM GVN has more features.

The following example, from [Value Numbering][], by Briggs, Cooper, and Simpson, illustrates how GVN removes redundant instructions across basic blocks:

```
void main(a0 : int, b0 : int, c0 : int, d0 : int, e0 : int, f0 : int) {
B1:
  u0 : int = add a0 b0;
  v0 : int = add c0 d0;
  w0 : int = add e0 f0;
  cond : bool = const true;
  br cond B2 B3;
B2:
  x0 : int = add c0 d0;
  y0 : int = add c0 d0;
  jmp B4;
B3:
  u1 : int = add a0 b0;
  x1 : int = add e0 f0;
  y1 : int = add e0 f0;
  jmp B4;
B4:
  u2 : int = phi u0 u1 B2 B3;
  x2 : int = phi x0 x1 B2 B3;
  y2 : int = phi y0 y1 B2 B3;
  z0 : int = add u2 y2;
  u3 : int = add a0 b0;
}
```

Our implementation removes the redundant additions in blocks `B2` and `B3`.
It also removes from block `B4`: 1) the meaningless phi node `u2 : int = phi u0 u1 B2 B3;`, which has arguments that are always equal, and 2) the redundant phi node `y2 : int = phi y0 y1 B2 B3;`, which is identical to the phi node that precedes it. 
Here is the full output, which matches the correct output in the paper:

```
void main(a0 : int, b0 : int, c0 : int, d0 : int, e0 : int, f0 : int) {
B1:
  u0: int = add a0 b0;
  v0: int = add c0 d0;
  w0: int = add e0 f0;
  cond: bool = const true;
  br cond B2 B3;
B2:
  jmp B4;
B3:
  jmp B4;
B4:
  x2: int = phi v0 w0 B2 B3;
  z0: int = add u0 x2;
  ret ;
}
```

### Copy propagation

```
int main(a : int) {
entry:
  b : int = id a;
  c : int = id b;
  d : int = id c;
  print d;
}
```

### Constant folding

```
int main() {
entry:
  a : int = const 1;
  b : int = const 2;
  c : int = add a b;
  print c;
}
```

2. Trying to get fewer instructions.

We compared the numbers of instructions in the original Bril programs in SSA form and in the same programs after running GVN and trivial dead code elimination (TDCE).

| Test program | Original | SSA form | After only LVN | After only GVN |
| :---        |    :----:   | :----: |  :----: |
| `test/gvn/constant-folding-tau-example.bril` | 26 | 37 | 37 | 35 |
| `test/gvn/constant-folding-calpoly-example.bril` | 27 | 25 | 25 | 22 |
| `test/gvn/add-commutativity.bril` | 6 | 6 | 6 | 5 |
| `test/gvn/copy-propagation.bril` | 5 | 6 | 6 | 3 |
| `test/gvn/constant_propagation.bril` | 5 | 6 | 6 | 4 |
| `test/gvn/across-basic-blocks.bril` | 8 | 8 | 8 | 7 |


3. Bigger benchmarks with TypeScript frontend and comparison with LVN

Difficulty:
- The code generated by the TypeScript frontend sometimes inserts `jmp` instructions after returns in the same basic block. We had to manually remove these in order for SSA conversion and dominator tree creation to work.

4. LLVM GVN tests that use more features than Bril has.

GVN provides the base for a number of additional optimizations, most of which we have not yet implemented.

[value numbering]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.36.8877&rep=rep1&type=pdf

