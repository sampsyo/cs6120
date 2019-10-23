+++
title = "Induction Variable Optimizations"
extra.author = "Daniel Weber & Drew Zagieboylo"
extra.bio = """
  [Daniel Weber](https://github.com/Dan12) is an MEng student interested in programming languages, distributed systems, and the inner workings of computers.
  [Drew Zagieboylo](https://www.cs.cornell.edu/~dzag/) is a 3rd year PhD student researching Security, Hardware Design, and Programming Languages. He enjoys rock climbing and gaming in his free time.
"""
+++

# Induction Variables

Loops are well known targets for optimization since they execute repeatedly
and significant execution time is spent in loop bodies.
The class of loop optimizations which we're considering in this post
are centered on special variables called _induction variables_ (IVs).
An induction variable is any variable whose value can be represented as a function of:
loop invariants; the number of loop iterations that have executed; and other induction variables.

Generally speaking, most induction variable optimizations are limited to
induction variables that are *linear functions* of their inputs.
For Bril, that means induction variables are computed only using
the `mul`, `add` and [`ptradd`](../manually-managed-memory) instructions.

# Optimization Overview

There are a large number of induction variable optimizations
which all have slightly different goals. Here, we're going
to give a brief overview on some of the optimizations we
implemented and what they're meant to achieve.

### Strength Reduction

In reality (despite what many software developers like to think),
not all instructions are really created equal. Some instructions
are more expensive to execute at runtime than others. For instance,
integer addition is usually "cheaper" than integer multiplication.
Induction variable strength reduction lets us "reduce" multiplication
operations on IVs to addition operations.

Take this simple program as an example:
```C
int j = 0;
for (int i = 0; i < 100; i++) {
    j = 2*i;
}
return j;
```

`j` is an induction variable dervied by applying a multiplication
to another IV, `i`. This makes it a perfect candidate for strength
reduction. Each iteration we set `j` to a brand new value
computed with that multiplication. Instead, every iteration we can increment `j`
by two times whatever we increment `i` by.

To simplify this optimization this is usually done by introducing a new variable
to represent the `2*i` value for each iteration.
```C
int j = 0;
int s = 0; //2*i when i == 0
for (int i = 0; i < 100; i++) {
  j = s;
  s = s + 2;
}
```
After [some other common compiler optimizations](https://en.wikipedia.org/wiki/Copy_propagation),
we can get this simpler version:
```C
int j = 0;
for (int i = 0; i < 100; i++) {
  j = j + 2; //+2 since i gets incremented by 1 each iteration
}
return j;
```

It's important to note that `j` no longer has a direct dependence on `i`
since there are no instructions which read from `i` and write to `j`.
Strength reduction often helps remove data dependencies, paving
the way for other IV optimizations.

### Induction Variable Elimination

```C
...
int max = 10;
int result = 0;
for (int i = 0; i < max; i++) {
    result++;
}
return result;
```

Simplifies to:

```C
int max = 10;
int result = 0;
for (; result < max; result++) {}
return result;
```


# Implementing Induction Variable Optimizations

It turns out that IV analysis require a large number 
of other static analyses before even thinking about optimization.

### Finding Loops

For instance, IV optimizations are all loop optimizations, which
means we need to identify loops. Natural loops are denoted by sets
of basic blocks that all have a common entry point *and* a "backedge"
in the control flow graph. This backedge corresponds to a branch or
jump in the CFG that goes back to the beginning of the loop.
Finding loops requires finding backedges, which it turns out
requires calculating dominators. A backedge is defined as
any edge in the control flow graph where the source vertex
_is dominated by_ the sink. Therefore to even start thinking about
optimizing we need to calculate the dominators and do a brief
reachability analysis. See the pictures below for an example CFG
with backedge annoations.

<img src="cfg.png" style="width:50%"/><img src="dom.png" style="width:50%"/>
On the left hand side we have the control flow graph where its only backedge
is represented as a dashed line. The right hand side picture shows all of the
dominators; each red line means "source is dominated by the sink". As you can see,
the only edge in the CFG which is the reverse of an edge in the dominator graph
is the backedge from `body` to `loop`.

There are some other subtleties here with nested loops or two loops which happen
to have the same entry block, but we ignore them in our implementation.
Our implementation correctness should be correct regardless of these subtleties
but may not be as optimal.

### Useful Dataflow Analyses



# Evalutaing our Optimizations