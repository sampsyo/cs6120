+++
title = "Making LLVM Address Calculation Safe(r)"
extra.author = "Drew Zagieboylo"
extra.bio = """
  [Drew Zagieboylo](https://www.cs.cornell.edu/~dzag/) is a 3rd year PhD student researching Security, Hardware Design, and Programming Languages. He enjoys rock climbing and gaming in his free time.
"""
+++

### LLVM Address Calculation

When compiling high level array and struct access to LLVM code, compilers
generally use the [getelementptr](https://llvm.org/docs/GetElementPtr.html)
(or GEP) instruction to calculate offsets into these memory allocations.
GEP instructions have the nice property that they are type aware; offsets are phrased
in terms of "number of elements" rather than "number of bytes."
For example, the code in this example dereferences some memory in
the middle of a struct (specifically the last element in the `b` field).

```C
struct EX {
  int a;
  char b[3];
  int *c;
}

struct *X;
...
return X->b[2];
```

In LLVM, we could write a single GEP instruction to calculate
the correct offset into the struct and then execute a `load` instruction
to actually dereference the pointer.

```C
%1 = getelementptr %struct.EX,  %struct.EX* %X, i64 0, i32 1, i64 2
%2 = load i8, i8* %1
ret %2
```

`%X` is our pointer argument, i.e. the base for our address calcuation.
Every following integral argument is an index into the prior datastructure.
The `i64 0` says that we want the *first* `struct EX` pointed to by `%X`.
If `%X` represented an array of structs, then this argument would tell us
which element of that array to access.

`i32 1` says "get the second element from struct EX," which is the `b` field.
Lastly, since `b` is an array of three elements, `i64 2` indicates
that the calculated address should point to the last element in that array.

The backend implementations of GEP will have to generate a sequence of actual
arithmetic operations based on the backend's representations of LLVM datatypes
to compile this instruction to real ISAs. However, most of the LLVM infrastructure
can still reason about pointer arithmetic without knowing these concrete representations
via these GEP semantics.

### Make GEP Safe

In general, 