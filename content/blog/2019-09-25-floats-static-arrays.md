+++
title = "Floating Points and Fixed-Length Arrays in Bril"
extra.author = "Dietrich Geisler"
extra.bio = """
  [Dietrich Geisler](https://www.cs.cornell.edu/~dgeisler/) is a 3rd year PhD student researching Language Design and Compilers.  Enjoys gaming and climbing.
"""
+++

### Introduction

My goal for this project was to add floating points and fixed-length arrays to [Bril](https://github.com/sampsyo/bril).  The intention behind this decision is to promote Bril as a lower-level intermediate language, with fewer type abstractions than Javascript without the complexity of representation of LLVM.  The hope is that, by writing the Braid IR to provide a _minimal_ but _descriptive_ set of numeric operations, users will be able to explore optimiziations without sacrificing visibility of low-level computational operations.

These goals were mostly successful at the level of Bril semantics, parsing, and interpretation.  What was more difficult than expected (and less successful) were my attempts to model TypeScript values and arrays with [TS2Bril](https://github.com/sampsyo/bril/blob/master/bril-ts/ts2bril.ts).  Before exploring the limitations of the Bril representation, however, we must examine the floating point and array semantics of Bril more formally.

### Floating Points in Bril

Bril now supports the types `double` and `float`.  These represent, respectively,  the stadard IEEE-754 [double-precision](https://en.wikipedia.org/wiki/Double-precision_floating-point_format) and [single-precision](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) floating-point formats.  These types come equipped with the basic arithmetic operations `fadd`, `fsub`, `fmult`, and `fdiv`, along with the comparitives `feq`, `flt`, `fgt`, `fle`, and `fge`.  The introduction of these operations is intended to highlight that floating-point and integer operations should be treated as fundamentally different objects.

In Bril, floating-point precisions should not be mixed.  It is expected, for instance, that the following code will produce a static exception:
```
v0: double = const 5;
v1: float = const 5;
v2: float = fadd v0 v1
```
Similarly, mixing floating-points and integers in Bril is undefined, as well as using floating-point arithmetic operations on integers or vice-versa.  In a future project, I intend to explore the consequences of relaxing these semantic requirements, particularly when mixing floating-point precision.

### Bril Fixed-Length Arrays

Bril now supports types of the form `array=type[size]`, where `type = | int | bool | float | double | array`.  Note the inclusion of type recursion in this definition, which permits multi-dimensional arrays.  Arrays in Bril must contain elements all of the same type, and the size of arrays cannot be changed after initialization.

There are three operations related to manipulating arrays in Bril: initialization, assignment, and indexing.  Arrays are initialized with the `new` instruction, which takes a type and stores a default initialization of that type to the given destination.  Array elements can be set with the `set` instruction, an effectful instruction which takes the array, an index, and a value in that order.  Finally, array elements can be read using the `index` instruction, which writes a value of the given array element.

Array operations are concisely summarized in the following code snippet:
```
val: int = const 5;
ind: int = const 1;

arr: int[3] = new int[3];   // arr = [0, 0, 0]
set arr ind val;            // arr = [0, 5, 0]
res: int = index arr ind;   // res = 5
```
Arrays can be written to freely in any Bril block.  As a result, array values are inherently stateful and should be treated by users carefully.  Writing values of the wrong type, however, is expected to not be permitted.

### Implementation

Floating points and fixed-length arrays have been implemented as described, and can be reasoned about by the `bril2json` parser, the `Brili` interpreter, and the `bril2txt` translator.  A subset of TypeScript can be written to Bril-recognized JSON by the `ts2bril` compiler.  While these conversions work generally as described above, there are some features which bear special notice.

The `Brili` interpreter supports integers, doubles, and floating-points all as different objects.  Integers are represented as Javascript [BigInt](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/BigInt) objects.  This choice of representation means that `Brili` does not correctly reason about integer overflow -- while this should be implemented in the future, I have not observed a need for it in the current Bril projects.

Doubles can be represented simply as `number`, while floats are derived with the [Math.fround](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/fround) function.  Note that `fround` is only applied _after_ each floating-point operation; this is permitted due to the [IEEE-754 restrictions](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) on floating point operation precision loss (or lack of).  Extending Bril to support other mathematical operations (such as `pow` or `sqrt`), may require that this implementation choice be updated.

TypeScript represents every number as a `double` by default, thus operations in TypeScript are translated to Bril as floating-point operations.  The TypeScript `bigint` type can be used to specially generate integers in Bril.  Operations between TypeScript `bigint` values are not supported at the time of this writing; however, this should be changed shortly.  Floating points cannot be easily represented by TypeScript code; however, I will look into using the `fround` function as a mechanism for generating Bril `float` values.

Fixed-length arrays are reasoned about naively by the `Brili` interpreter; in particular, it makes no attempt to check array length before indexing.  `Brili` arrays are filled with 0s by default when initialized.

`ts2bril` can compile TypeScript arrays of consistent types to a `Bril`-like language.  This language differs from Bril only in that fixed-length arrays may have a size of `-1`, indicating that the size of the array is not known.  I am planning to explore ways to fix this issue; as TypeScript does not track array length and the Bril semantics for arrays is extremely limited, it should be very straightforward to take a second pass after compiling to replace each instance of unknown array length with the correct size.

I found working with `ts2bril` to be _extremely_ painful when adding arrays, and spent far too long trying to grapple with the TypeScript type system.  For me, this project highlighted the lack of internal documentation for TypeScript type structure.

### Evaluation

Testing was focused on correctly interpreting several new files added to the Bril [test suite](https://github.com/Checkmate50/bril/tree/arrays/test).  These tests all work as intended; however, it is worth noting that the TypeScript array interpretation file was simplified somewhat as the limitations of the compiler became clear.  Speed of implementation was not a concern in this project.

Some evaluation details of note include floating-point exactness testing and TypeScript compilation limitations.  Some simple floating-point operations [provided](https://github.com/Checkmate50/bril/blob/master/test/interp/float.bril) to `Brili` were compared to a similar C implementation ([compiled online](https://www.onlinegdb.com/online_c_compiler)), and found to be identical in precision.  
I originally intended to verify TypeScript compilation by interpreting the resulting code; that is, by running the command:
`ts2bril src.ts | bril2txt | bril2json | brili`
Due to the limitations in the `ts2bril` compiler described above, however, I was unable to achieve this goal with my TypeScript [array test file](https://github.com/Checkmate50/bril/blob/arrays/test/ts/array.ts).