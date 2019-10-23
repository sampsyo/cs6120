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

A bundle is defined as an arbitrarily long sequence of conditions, a sequence
of four instructions, and a label.

```
[ c1, c2, c3 ...; i1, i2, i3, i4; label ]
```

The semantics of the bundle is if `c1 && c2 && ...` is true, execute the
instructions, otherwise jump to the label. In a real machine

## Superblock Scheduling

### Compaction

### List Scheduling

### Trace Scheduling


## Implementation

## Evaluation

### Cost Model

### Comparison to Compaction
