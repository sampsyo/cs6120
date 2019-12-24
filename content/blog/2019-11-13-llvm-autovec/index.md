+++
title = "LLVM Loop Autovectorization"
[extra]
bio = """
  Gabriela is a graduate student researching computational imaging at the small scale.
  Ryan is a PhD student at Cornell. He studies programming languages
  and verification.
  Rolph is a third-year graduate student studying the intersection of
  programming languages, security, and distributed systems.
"""
latex = true
[[extra.authors]]
name = "Gabriela Calinao Correa"
[[extra.authors]]
name = "Ryan Doenges"
link = "https://ryandoeng.es"
[[extra.authors]]
name = "Rolph Recto"
link = "https://rolph-recto.github.io"
+++

We can use SIMD support in modern processors to optimize programs.
Specifically, a compiler can automatically detect when loops can be
*autovectorized* so that multiple iterations of the loop can be performed
at once.
We implement an autovectorizer pass for LLVM that performs this optimization
for LLVM IR. 
LLVM natively supports vector instructions already, so the pass was implemented
with relatively few lines of code by taking advantage of LLVM's existing
infrastructure.

The repository can be found [here](https://github.com/rolph-recto/cs6120-autovec).


## Design Overview

We implemented our vectorizer as a transformation pass in LLVM.
We overrode the `LoopPass` base class, which provides a `runOnLoop` method
that child classes can override, running on all the natural loops found
for a particular method.

Our vectorizer pass comes in two stages.
First, it checks whether a loop can be vectorized at all.
Then, it vectorizes instructions in the loop and updates the loop stride to
match the vector size.

We deem a loop vectorizable if it satisfies the following criteria:

* The loop is in canonical form: it has an inductive variable (read: a unique
  variable that enters the loop at 0 and increments by 1 per iteration)
  and it has a single block from which the loop exits.

* The loop bound is a constant and is divisible by the vector size.
  We determine the loop bound by checking the condition on which the unique
  exiting block in the loop exits.

* The loop has no cross-iteration dependencies.
  Vectorization assumes that adjacent loop iterations can be parallelized,
  which is not true if the iterations have dependencies on each other.
  To check that cross-iteration dependencies do not exist, 
  our vectorizer checks that all array accesses are indexed either by the
  inductive variable or loop-invariant data.
  It also checks that operands for all operations in the loop either are
  (1) vectorizable, (2) loop-invariant, or (3) the induction variable.
  Branches in the loop are checked to see if their condition, if they have one,
  is loop-invariant.
  This ensures that the loop does not vary in control flow between adjacent
  iterations.

Once we have deemed a loop vectorizable, we vectorize instructions by changing
the types of instructions into their vectorized counterparts.
This is possible because vector types in LLVM are first-class and are treated
like other types like `int`.

For `store`s, `load`s, and operation instructions (e.g., `add`, `mul`, `icmp`),
we replace the operands with their vectorized counterparts.
For constant operands, these are vectorized in place; otherwise, we create
a vectorized version of the operand immediately after its definition site
using `insertelement` instructions, and then replace uses of the operand
with its new vectorized definition.
There are two possible cases for operands:
(1) the operand is loop-invariant, in which case the vector contains `n` copies
of the original operand given where `n` is the vector size;
or (2) the operand is the inductive variable, in which case
(assuming the inductive variable is `i`) the vector is of the form
`<i, i+1, ..., i+(n-1)>`.


For `getelementptr` instructions, there are only two possible cases given
our vectorization conditions above:
(1) the base pointer is indexed by loop-invariant data, in which case
the vectorizer does nothing;
or (2) the base pointer is indexed in at least one position by the
inductive variable, in which case we immediately `bitcast` the result of the GEP
into a vector type.
For example, if the GEP returns `uint8_t*`, we bitcast the resulting pointer
into vector type `<n x uint8_t>*`, where `n` is the vector size.
We then replace all the uses of the GEP result with its `bitcast`ed counterpart.
Thus `load`s from the pointer load not just data from the array index, but
its adjacent indices as well, thus loading a vector from memory locations
and not just a single datum.
`store`s using GEP results can also write vectors to memory in this way.

Finally, once all instructions are vectorized, we find the unique instruction
that increments the inductive variable and change its stride to match the
vector size.
(Given our vectorization check, we can assume the inductive variable is a
`PHINode` with two incoming definitions: a constant `0` and an addition
instruction that increments the inductive variable.
This is how we find the inductive variable's unique update instruction.)


## Example

To show our vectorizer in action, consider the following example:

```C
  int64_t a[100];
  for (int64_t i = 0; i < 100; i++) {
    a[i] = i;
  }
```

This C code compiled to LLVM IR[^mem2reg] looks like:

[^mem2reg]: The IR generated has been transformed into SSA form using the
`mem2reg` pass.

```
entry:
  %a = alloca [100 x i64], align 16
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i64 %i.0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr [i64]* %a, i64 0, i64 %i.0
  store i64 %i.0, i64* %arrayidx, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret i32 0
```

Running our vectorization pass, we get the following:

```
entry:
  %a = alloca [100 x i64], align 16
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %0 = insertelement <4 x i64> zeroinitializer, i64 %i.0, i64 0
  %1 = insertelement <4 x i64> %0, i64 %i.0, i64 1
  %2 = insertelement <4 x i64> %1, i64 %i.0, i64 2
  %3 = insertelement <4 x i64> %2, i64 %i.0, i64 3
  %4 = add <4 x i64> %3, <i64 0, i64 1, i64 2, i64 3>
  %cmp = icmp slt i64 %i.0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr [i64]* %a, i64 0, i64 %i.0
  %5 = bitcast i64* %arrayidx to <4 x i64>*
  store <4 x i64> %4, <4 x i64>* %5, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i64 %i.0, 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret i32 0
```

The vectorizer changes the loop so that each new iteration performs 4 iterations
of the original loop.
The GEP in the `for.body` block computes the base address for a 4-vector;
the subsequent `store` into `%arrayidx` then uses the vectorized version 
of the inductive variable `%i.0` computed in `for.cond` (`%4`)
to write into memory the next 4 values of `%i.0` at once.


## Evaluation

We evaluated our autovectorizer across a range of small benchmarks containing
loops.
We ran experiments on a ThinkPad T410 with an Intel Core i7-620M processor and
8GB of RAM.
We constructed our benchmarks so that all loops can be vectorized according
to the rather stringent vectorization conditions, which disallows indexing
arrays with anything other than the inductive variable or loop-invariant data.

The results below are measured across three runs.
Note that DEF is the configuration without any optimizations (`-O0`),
while OPT is the configuration with only our autovectorizer enabled.

Benchmark   | DEF runtime (ms) | OPT runtime (ms) 
------------|------------------|------------------
test1       | 26.00            | 13.00            
test2       | 25.00            | 22.67            
test3       | 19.00            | 15.33            
test4       | 25.33            | 15.00            
test5       | 10.00            | 11.00

Across the benchmarks, either OPT outperforms DEF handily
(as in `test1` and `test4`), or it runs about the same time
as its DEF counterpart.
This is the behavior we expected.

It is not immediately clear what makes `test1` and `test4` different from the
other testcases.
They are the only testcases with both of the following features:
division expressions and a single, non-nested loop.
The rest of the testcases either have multiple loops, nested loops, or no
division expressions.
More investigation is needed to determine the exact reason why the other
vectorized testcases do not run noticeable faster than their
non-vectorized counterparts.

