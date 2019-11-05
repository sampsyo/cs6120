+++
title = "A Unified Theory of Garbage Collection"
extra.author = "Mark Anastos"
extra.author_link = "https://github.com/anastos"
extra.bio = """
  Mark Anastos is an undergraduate senior studying Computer Science and
  Electrical & Computer Engineering.
"""
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

fixed-point formulation
minimal vs maximal fixed point -- cycles
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
