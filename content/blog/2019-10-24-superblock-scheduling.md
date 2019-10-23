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
machine, there are either bundles that contain up to 4 instructions or a single
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
List scheduling is a simple heuristic based algorithm that takes a list of
instructions and returns a list of VLIW instructions. We start by presenting
a high level overview of the algorithm and then we will go into more detail
about each part. We first build a dataflow graph
that represents the dependencies between instructions. Then a heuristic assigns
each instruction in the graph a priority. We initialize a queue with all the instructions
that have no predecessors in the graph. Then, until we have scheduled all the instructions,
we do the following in a loop:
- Form an empty VLIW instruction
- Take the instruction from the queue with the highest priority
- Add it to the VLIW instruction if compatible
- Continue taking instructions from the queue until either the queue is empty or the instruction 
is incompatible with the current VLIW instruction.
- For each element that we scheduled, check if their successors now have all their predecessors scheduled
and add them to the queue

The important property of this algorithm is that it doesn't schedule an instruction until all
of it's predecessors in the DAG have been scheduled. In order to maintain program correctness,
we have to make sure that the predecessor relationship in the DAG respects data dependencies
between instructions.

If we were just considering basic blocks, we can do this just by looking at what an instruction
reads and writes. Let `i`, `j` be two instructions such that `i` comes before `j`. If `i` reads from
variable `x` and `j` writes to `x`, then we never want to reorder `j` before `i`. We can encode this by
making `i` a predecessor of `j` in the DAG. Similarly, if `i` writes to some variable `x`, 
and `j` reads from `x`, then we want to make sure that `j` is never scheduled before `i` so that `j`
will see the writes to `x`. We can again encode this by making `i` a predecessor of `j`.

This would be enough for basic blocks. However, traces in general can have multiple basic blocks
and they can contain branch instructions (although only one branch body will be in the trace).
Importantly, this means that it is possible to exit from the middle of a trace. Because of this,
we have to be careful about what we move above potential exit points. Consider the following trace fragment:
```
  ...
  v6: bool = gt v4 v5;
  br v6 for.body.2 for.end.2;
for.body.2:
  v7: int = id result;
  v8: int = id i;
  v9: int = mul v7 v8;
  result: int = id v9;
  ...
```
When we consider this as a trace, we ignore the label and assume that the branch will jump to `for.body.2`.
If we just use the dependency rules discussed above, the following is a valid reordering:
```
  ...
  v6: bool = gt v4 v5;
  v7: int = id result;
  v8: int = id i;
  v9: int = mul v7 v8;
  result: int = id v9;
  br v6 for.body.2 for.end.2;
  ...
```
BAD BAD BAD BAD  
It's possible that the assumption that we jumped to `for.body.2` is false and we have to jump to `for.end.2`.
If this happens, then the `for.end.2` block will see writes to `result` that it shouldn't have.

To fix this, we say that a `br` instruction reads everything that is live below it. This prevents writes to
live variables from being moved above the `br` instruction.

The other important ingredient in this algorithm is the heuristic that assigns priorities to instructions.
We follow [Fisher's](https://people.eecs.berkeley.edu/~kubitron/courses/cs252-S12/handouts/papers/TraceScheduling.pdf)
example and use the **highest levels first** heuristic. The priority of each node is the depth of the longest chain
in the dependency DAG. He claim's close to "optimal in practical environments" with this heuristic.
Given more time, it would be interesting to explore how changing this heuristic effects the results
of trace scheduling.

### Trace Scheduling


## Implementation

## Evaluation

### Cost Model

### Comparison to Compaction
