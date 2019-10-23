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

A bundle is defined as sequence of conditions, a sequence
of four instructions, and a label.

```
[ c1, c2, c3 ...; i1, i2, i3, i4; label ]
```

The semantics of the bundle is if `c1 && c2 && ...` is true, execute the
instructions, otherwise jump to the label. In hardware, this can be implemented
by guarding the write-back stage with the value of the conditional.

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
`v4` in parallel. If `v7` is false, the program switches back to normal
execution, computing the `v1` and `v4` sequentially. A superblock compiler
might choose to make various part of the "slow" program paths traces
themselves, trading off program size for speed.

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

For our project, we implemented four things:

- **Bril Extension**: We extended the Bril interpreter by adding a "group"
   instruction which has the semantics described above. The interpreter simply
   executes the instructions in the group if the conditionals are true and
   otherwise jumps to the label. We also add a "bundle counter" to track
   the number of instructions executed in the interpreter. Each bundle and
   instruction counts as one.

- **Implement control and data flow analysis**: We implemented a passes to generate
  the CFG of the Bril program and a straightforward live variable analysis.
  Both of these analysis are used by the Superblock scheduling algorithm to
  correctly incorporate branches into a trace.

- **Superblock scheduling**: **TODO(sam)**

- **Generating valid programs from traces**: Once we have a program trace, we
  change the programs to correctly jump into and out of traces. We also duplicate
  the code in the trace to correctly work for the slow program path.

## Evaluation

### Cost Model

### Comparison to Compaction

## Conclusion

Superblock scheduling is an optimization that helps VLIW compilers generate
larger program traces thereby allowing for faster execution in the common case.
The underlying philosophy behind this optimization has been widely adopted by
moder JITs that dynamically generate specialized fast code for program paths
that get executed in the common case.

Trace Scheduling, a more general version of superblock scheduling that allows
traces to have multiple inputs and outputs was also studied as a part of
efforts to design VLIW machines. Specifically, the [ELI-512](https://courses.cs.washington.edu/courses/cse548/16wi/Fisher-VLIW.pdf) was built with
special support to execute programs generated by its trace scheduling compiler
[Bulldog](https://dl.acm.org/citation.cfm?id=912347).
