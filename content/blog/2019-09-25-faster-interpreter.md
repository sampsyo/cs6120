+++
title = "Writing a faster interpreter for Bril"
extra.bio = """
Daniel Glus is a Cornell undergraduate in CS.

[Wil Thomason](https://www.cs.cornell.edu/~wil) is a
nth year Cornell PhD student in CS. He primarily works on planning algorithms for robotics.
"""
extra.author = "Wil Thomason and Daniel Glus"
+++

# Interpreting Bril

[Bril](https://github.com/sampsyo/bril) is a pleasantly simple intermediate representation for
teaching compilers and compiler optimizations. Part of Bril being useful for this purpose is the
existence of a solid, simple reference interpreter for the language: `brili`. Written in
[TypeScript](https://www.typescriptlang.org/), `brili` is straightforward to extend and allows easy
experimentation with new language features and optimizations. The flipside of `brili`'s simplicity
and the use of TypeScript for its implementation is that is isn't very fast. Thus, while `brili` is
suitable for working with small, simple Bril programs, it may make experiments using more complex,
long-running programs unnecessarily onerous.

# Gotta go fast

Given this, our goal was to make a faster interpreter for Bril. The simplest way to do this is to
pick a faster language than TypeScript for the interpreter's implementation. As such, we chose to
[Rewrite It In Rust](https://github.com/ansuz/RIIR). [Rust](https://www.rust-lang.org/) is a
modern systems programming language designed to make it easy to "build reliable and efficient
software"; it typically performs comparably to (or sometimes faster than) the same software written
in a more traditional systems language such as C or C++.

Rust is appealing for this project for a few other reasons. Chief among these is its rich library
ecosystem, which includes the excellent [Serde](https://serde.rs/). Serde is a library (technically,
a family of libraries) for **ser**ializing and **de**serializing data between a range of formats,
including JSON. It provides utilities for automatically [deriving](https://serde.rs/derive.html)
functions to parse JSON (and other formats) directly into native Rust structs, which is convenient
for an interpreter. With Serde, we can define the data structures for Bril's various operations and
overall program structure, then get parsing code from Bril's JSON representation for free! Even
better, Serde's automatically generated code is typically very fast.

# Interpreter structure

Our first "draft" of an interpreter implemented a simple structure:
1. Using Serde, derive JSON deserialization code for types representing the Bril operations
and other syntax (including labels, functions, etc.)

2. Parse Bril JSON into the aforementioned structures.

3. Form basic blocks and a CFG from the program instructions. This process constructs an mapping
from label names to basic block indices and uses this map to "link" together connected basic blocks.
Linking in this way (rather than just constructing the name/index mapping and using that at runtime)
lets us have slightly faster execution - it's faster to load an array index than it is to e.g.
load a heap-allocated object (as in a pointer-linked graph) or compute a string hash every time we
need to find a label location. It's worth noting that while we could've gotten away with not forming
basic blocks or a CFG, and potentially gained some speedup from this laziness, the CFG building
process should not dominate interpretation times and enables us to build more optimizations down the
line.

4. Starting at the first basic block of the `main` function, iterate through, matching each
instruction based on its type (using Rust's `match` expression on a large variant type representing
the Bril operations). Each branch of this match implements the semantics for a Bril operation; we
also handle type-checking at this stage. We handle control flow by setting the index of the "next"
basic block for the next iteration of the main execution loop.

# `#[derive(Problems)]`

The above is a perfectly reasonable but naive implementation of a Bril interpreter. There are a lot
of possible improvements we could make on this baseline; many of these stem from being smarter with
how we look up variable values and how we dispatch operations. To enable these sorts of
improvements, it helps to be able to parse variable identifiers and operation types into numerical
representations (to enable direct indexing instead of name hashing for variable lookup, and to
enable branch-predictor friendly dispatch of operations based on broad classes of operation "type").
We should note, however, that this transformation is not free: for short programs where names are
only accessed once/a handful of times, it may cost more to perform identifier transformation before
interpretation. This transformation will be most beneficial for programs with loops and frequent
variable access.

We tried to implement this in a relatively clean way around the original Serde structure, but ran
into some challenges that proved insurmountable in the time we had for this assignment: the
difficulty of performing stateful deserialization and fighting the borrow checker about deserialized
type lifetimes.

## State of the Deserialization

Here's our basic approach to replacing variable identifiers with indices, in pseudocode (Python):
```python
next_id = 0
id_name_map = {}
def deserialize_identifier(ident):
  if ident in id_name_map:
    return id_name_map[ident]

  id_name_map[ident] = next_id
  next_id += 1
  return next_id - 1
```

We ran into two problems implementing this in Rust. First, Rust is strongly typed. This means that
either we need to perform this transformation during deserialization of JSON into Rust data
structures, or we need two versions of our Rust structures: One which represents identifiers as
strings, and one which uses the above numerical representation.

The first option here is problematic because Serde makes it challenging to use mutable state inside
the deserialization logic for a deeply-nested field of a data type. Indeed, it seems that you lose
most of the benefits of Serde's `#[derive(Deserialize)]` magic auto-implementation, and have to
implement a tree of deserializers manually.

The second option seems better: make the IR data types polymorphic and deserialize to a `String`
specialization of the types, then run a pass over the program to transform to a numerical
specialization of the types. It's easy to use mutable state in this second transformation, and the
use of parametric polymorphism makes the code clean.

When we tried this, we made a version of the interpreter that compiled, but failed to deserialize
JSON correctly. Though Serde is usually able to automatically derive deserialization logic for
generic types, it seems to have done something unintended for us here - we're still not entirely
sure why.

# How do we stack up?

**TODO:** Eval results here

# What else could be done?

If we continue to develop the interpreter, it would be worthwhile to try and overhaul our parsing
logic to enable deserializing variable identifiers and operations into numerical representations.
This would enable us to try adding some interpreter optimizations (e.g. those found
[here](https://github.com/status-im/nimbus/wiki/Interpreter-optimization-resources)) to squeeze out
more speed. It isn't immediately clear how to best do this, given the challenges we encountered in
attempting it. One (fairly hacky) approach could be to use macros to generate poor-man's generics: A
copy of each IR type for `String` (for deserialization) and a version for indices/numerical
representations (for interpretation). This would increase code size (though so do generics) and
feels like a brittle solution. Another option is to use `unsafe` features and global mutable state
to perform the transformation between strings and numerical representations during deserialization;
this could work, but feels like an abuse of the language. Ideally, there is a way to get Serde to
implement deserialization logic only for the string-based generic specialization of the IR types,
but it's not clear if this is possible. 

Aside from adding interpreter implementation optimizations, it would also be interesting to apply
CFG-level optimizations and compile Bril to some optimized in-memory bytecode representation for
faster interpretation, as well as to add some of the cool new language features created as a part of
other Project 1's.
