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
and the use of TypeScript for its implementation is that it isn't very fast. Thus, while `brili` is
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

We implemented this in a relatively clean way around the original Serde structure, but ran into some
challenges around the difficulty of performing stateful deserialization.

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

The second option is better: make the IR data types polymorphic and deserialize to a `String`
specialization of the types, then run a pass over the program to transform to a numerical
specialization of the types. It's easy to use mutable state in this second transformation, and the
use of parametric polymorphism makes the code clean.

We've implemented this, and it is working. Most of the potential speedup remains untapped - we did
not have time in this project to implement parsing of operations into branch-friendly numerical
representations.

# How do we stack up?

We ran benchmarks using [hyperfine](https://github.com/sharkdp/hyperfine). The same Bril program was
run using both `brili` and `brilirs`. Overall, `brilirs` was faster, as expected. On one benchmark, `brili`
was twice as fast, however. The results (measurements reported as mean plus/minus standard
deviation):

| Benchmark        | brili            | brilirs           | Speedup                            |
|------------------|------------------|-------------------|------------------------------------|
| matrix_mul, n=10 | 45.5 ms ± 3.4 ms | 21.0 ms ± 2.7 ms  | 2.16 ± 0.32                        |
| matrix_mul, n=20 | 82.2 ms ± 2.5 ms | 148.9 ms ± 3.6 ms | brili was 1.81 ± 0.07 times faster |  
| poly_mul, n=50   | 53.4 ms ± 3.9 ms | 44.2 ms ± 1.9 ms  | 1.21 ± 0.10                        |
| poly_mul, n=100  | 86.2 ms ± 5.0 ms | 174.7 ms ± 4.0 ms | brili was 2.03 ± 0.13 times faster |

For other benchmarks, `brilirs` was so fast that hyperfine warned that the average run was under or
around five milliseconds:

| Benchmark       | brili            | brilirs         | Speedup       |
|-----------------|------------------|-----------------|---------------|
| factorial       | 38.2 ms ± 4.2 ms | 2.4 ms ± 1.3 ms | 16.09 ± 8.84  |
| fibonacci        | 39.8 ms ± 3.3 ms | 4.1 ms ± 2.4 ms | 9.74 ± 5.68   |
| id_chain, n=10  | 36.8 ms ± 3.0 ms | 1.9 ms ± 1.1 ms | 19.06 ± 11.06 |
| id_chain, n=500 | 39.4 ms ± 4.1 ms | 5.7 ms ± 1.2 ms | 6.89 ± 1.64   |
| poly_mul, n=10  | 37.8 ms ± 2.3 ms | 5.6 ms ± 1.3 ms | 6.80 ± 1.65   |

Most of the benchmarks are from [bril-benchmark](https://github.com/xu3kev/bril-benchmark/).

Benchmarks were run on a 2018 Thinkpad T580 running Arch Linux, kernel version 5.2.11.

# What else could be done?

If we continue to develop the interpreter, it would be worthwhile to try adding some interpreter
optimizations (e.g., those found
[here](https://github.com/status-im/nimbus/wiki/Interpreter-optimization-resources)) to squeeze out
more speed.

Aside from adding interpreter implementation optimizations, it would also be interesting to apply
CFG-level optimizations and compile Bril to some optimized in-memory bytecode representation for
faster interpretation, as well as to add some of the cool new language features created as a part of
other Project 1's.
