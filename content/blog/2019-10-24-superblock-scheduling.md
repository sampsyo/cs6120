+++
title = "Superblock Scheduling for Bril"
extra.author = "Sam Thomas & Rachit Nigam"
extra.bio = """
  [Rachit Nigam](https://rachitnigam.com) is a second year PhD student interested in
  programming languages & computer architecture. In his free time, he
  [subtweets](https://twitter.com/notypes/status/1170037148290080771) his advisor and [annoys tenured professors](https://twitter.com/natefoster/status/1074401015565291520).

  Sam Thomas is a senior undergraduate student at Cornell. He is applying to Grad Schools for applied PL.
"""
+++

## A Very Long Instruction Word

Very Long Instruction Word (VLIW) architectures allow execution of multiple
instructions in the same cycle. Instead of executing a single instruction,
the architecture executes a "bundle" of independent instructions, allowing it
to harness Instruction Level Parallelism (ILP). The key trade-off VLIW
architectures make is pushing the instruction scheduling choices to the compiler.
While a modern out of order processor will discover and dynamically change
the execution order of a sequence of instructions, VLIW processors require
the compiler to explicitly schedule parallel operations into bundles.

The central challenge in designing VLIW compilers is minimizing the number of
`nop`s in bundles. A bundle has a `nop` when not enough parallel instructions
can be found to fit the bundle's size (say 3 instructions per bundle).
Read-write conflicts, control dependencies, and structural hazards (not having
sufficient number of hardware units to support a bundle's parallel execution)
force compilers to move instructions into separate bundles to preserve
sequential semantics.

Superblock scheduling is an optimization for VLIW compilers that allows them to
generate denser bundles while increase the program size. For our project, we
extended the Bril interpreter to emulate a VLIW machine and implement the
superblock scheduling optimization to generate bundles of instructions from
source Bril programs. Finally, we evaluated the effectiveness of our
optimization by measuring (1) program cost (measure by the number of bundles
executed) and (2) measuring how often we fall out of a superblock (a sequence
of dense bundles).

## An Imaginary VLIW machine

We extended Bril's interpreter to simulate a VLIW machine. In the Bril VLIW
machine, there are either bundles that contain 4 instructions or a single
instruction (equivalent to a bundle with that instruction and three `nop`s).

A bundle is defined as sequence of conditions, a sequence
of four instructions, and a label.

```
[ c1, c2, c3 ...; i1, i2, i3, i4; label ]
```

The semantics of the bundle is if `c1 && c2 && ...` is true, execute the
instructions, otherwise jump to the label. For real hardware, this would require
having a write buffer that tracked the changes from the instructions and throw
out the changes if the conditional computed in parallel was false.

## Superblock Scheduling

A straightforward implementation of VLIW compilation at the basic block can
simply try to perform code motion and bundling of instructions in a basic
block. Since there are no jumps or branches, the code can be easily compacted.
However, basic blocks tend to be small which leads to the compiler missing
out on optimization opportunities.

In the following code block, the computations for `v1` and `v4` can be
performed in parallel (inside a bundle). However, because of the branch instruction
between the two blocks, the compiler cannot move the two instructions into
a bundle.

```
b1:
  v1: int = add v2 v3
  br v7 b2 b3
b2:
  v4: int = add v2 v0
b3:
  ...
```

The core idea with superblock scheduling is finding "traces" of frequently
code by predicting which branch a program is going to take and building
a fast program path with dense bundles. In case the branch prediction is
wrong, the compiler adds abort labels to exit the trace.

With superblock scheduling, the code example above can be turned into:

```
b1:
  [ v7; v1: int = add v2 v3, v4: int = add v2 v0; slow ]
  br v7 b2 b3
slow:
  v1: int = add v2 v3
  br v7 b2 b3
b2:
  v4: int = add v2 v0
b3:
  ...
```

In the fast case (when `v7` is true), the program will compute both `v1` and
`v4` in parallel. If `v7` is false, the program switches back to normal execution,
computing the `v1` and `v4` sequentially. A superblock compiler might choose
to make various part of the "slow" program paths traces themselves, trading
off program size for speed.

### List Scheduling

### Trace Scheduling


## Implementation

## Evaluation

### Cost Model

### Comparison to Compaction
