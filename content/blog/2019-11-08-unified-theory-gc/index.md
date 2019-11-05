+++
title = "A Unified Theory of Garbage Collection"
extra.author = "Mark Anastos"
extra.author_link = "https://github.com/anastos"
extra.bio = """
  Mark Anastos is an undergraduate senior studying Computer Science and
  Electrical & Computer Engineering.
"""
extra.latex = true
+++

## Introduction

talk about unifying tracing and reference counting, which are typically thought
of as completely separate.

## Background

what is the point of garbage collection?
define tracing and reference counting
differences between them

## Intuition 

matter vs anti-matter
increasing vs decreasing

## Mathematical Model

In order to analyse and compare different garbage collection strategies, the
paper presents a mathematical model of the memory management problem that
garbage collection is trying to solve. Objects in memory and the pointers that
they contain are modeled as the nodes and edges of a graph. The set of objects
is denoted as $V$, and the multiset of pointers between objects is $E$. An
object should not be freed if it could be used in the future. A conservative
approximation of this, without any program analysis, is that an object could be
used in the future if there exists a path of pointers to the object which
originates from the stack or from a register. We call the starting points of
such paths (i.e., all objects to which there is a direct pointer on the stack or
in a register) the roots of the graph, which make up the multiset $R$.

Using these definitions, we can formulate the reference counts of objects
(denoted $\rho(v)$ for $v \in V$) as a fixed point of the following equation:

$$ \rho(v) = \big|[v : v \in R]\big| +
             \big|[(w, v) : (w, v) \in E \land \rho(w) > 0]\big| $$

Here we recursively define the reference count of an object $x$ to be the number
of root pointers to $x$ plus the number of pointers to $x$ from objects which
themselves have non-zero reference counts. Any object whose reference count is
zero according a fixed point of $\rho$ can be freed, as there is no way for the
program to reference it in the future.

However, it is important to note that there could be multiple fixed points to
this equation, namely in the presence of cyclic garbage. If object $a$ points to
object $b$, and vice versa, but neither is a root and neither is pointed to from
elsewhere, then, by this formulation, both $\rho(a) = \rho(b) = 1$ and $\rho(a)
= \rho(b) = 0$ are valid solutions. Ideally, a garbage collection algorithm will
find the least fixed point of $\rho$, meaning that it will consider all cyclic
garbage as able to be freed. A tracing collector does this, whereas a reference
counting collector does not detect any cyclic garbage, and thus finds the
greatest fixed point.

strategies for collecting cycles
    backup tracing
    trial deletion

## Tracing & Reference Counting are Duals

explain algorithms and how they are related/opposites

## Hybrids

### Deferred Reference Counting & Partial Tracing

### Generational Collectors

### Multi-Heap Collectors?

## Cost Analysis

define variables
general time and space formulas

## Conclusion

gc design strategies
    partitioning memory
    traversal
    trade-offs
