+++
title = "Making LLVM Address Calculation Safe(r)"
extra.author = "Drew Zagieboylo"
extra.bio = """
  [Drew Zagieboylo](https://www.cs.cornell.edu/~dzag/) is a 3rd year PhD student researching Security, Hardware Design, and Programming Languages. He enjoys rock climbing and gaming in his free time.
"""
+++

# Memory Safety in LLVM

LLVM IR code is not generally memory safe.
While certain *obviously bad* behaviors are
disallowed, it is not hard to write code that
may execute out-of-bounds memory accesses at runtime.

For instance, the size of an
array may be statically known,
but the access index may be unknown at compile time:

```C
int foo(int x) {
    int tmp[10];
    ...//some code to set values in tmp
    return tmp[x];
}
```
In this project we seek to improve the memory
safety of LLVM programs by inserting dynamic bounds
checks at runtime that cause the program to stop
executing rather than violate memory safety.
After running our compilation pass the aforementioned
code would have the following runtime behavior:

```C
int foo(int x) {
    int tmp[10];
    ...//some code to set values in tmp
    if (x >= 0 && x < 10) {
       return tmp[x];
    } else {
       exit(1);
    }
}
```
### LLVM Address Calculation

When compiling high level array and struct access to LLVM code, compilers
generally use the [getelementptr](https://llvm.org/docs/GetElementPtr.html)
(or GEP) instruction to calculate offsets into these memory allocations.
GEP instructions have the nice property that they are type aware; offsets are phrased
in terms of "number of elements" rather than "number of bytes."
For example, the code in this stub dereferences some memory in
the middle of a struct (specifically the last element of the `b` field).

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

This functionality makes GEP an ideal point for analyzing out-of-bounds accesses.
Before a program might make an out-of-bounds access it has to
acquire an out-of-bounds pointer. Usually, this means it
executes a GEP whose result will then later be the argument of a
load or store operation.

Our approach in this implementation is to prevent any
runtime GEP instructions from executing which might lead to
illegal memory accesses. If a program can never acquire
an out-of-bounds pointer, it can't violate memory safety.

### Making GEP Safe(r)

Let's go back to our first example of an out of bounds array access:

```C
int tmp[10];
...
return tmp[x];
```

The return statement roughly translates to:

```C
%addr = getelementptr [10 x i32], [10 x i32]* %tmp, i64 0, i64 %x;
%val = load i32, i32* %addr     ;
ret %addr;
```

In order to insert a dynamic check for memory safety, there
are two things we need to know:

 - What is actual access index value?
 - What are legal access index values?

Happily, when considering GEP instructions, the first question is easy
to answer; each operand represents an access index
value. We can dynamically insert instructions into the program that compare
those operands to other values.

The second question is a much more difficult problem,
whose subtleties we'll address in the next section.
For the most part though, this will be addressed by LLVM's type
information. Based on its type annotation, we know
 that `%tmp` points to an array with 10 32-bit integers.
Therefore, we can conclude that the only valid values for `%x` are
between 0 (inclusive) and 10 (exclusive).

In general our algorithm is this:

```
1) Initialize the current type to be the type of the first operand.
2) Initialize current operand to the first index operand.
3) If possible, insert instructions to check if current operand is in bounds based on current type
4) Set current type to the target type (e.g. if current type is `int [][]` set it to `int []`).
5) If there are no more index operands, exit.
   Else, set the current operand to the next in the operand list and goto (3).
```

### Pointer Sizes

In the above examples, we could always tell how big our memory allocations
were since they were allocated with static sizes. `int tmp[10]` comprises of
two static allocations: 1) a single pointer-sized memory cell (to contain the local variable `tmp`); 2) a memory cell
containing 10 integers (the memory pointed to by `tmp`).

In many cases, the sizes of arrays may be difficult or impossible to determine at compile
time. Consider the following snippet:

```C
int foo(int x, y) {
    int tmp[x];
    ... //init values in tmp
    return tmp[y];
}
```
In the corresponding LLVM code, the type of `tmp` is no longer a sized type;
it is just `i32*`. We can no longer use types to help us determine what
are and are not legal offsets. In this case, however, there is something
that we can do. LLVM uses [`alloca`](https://llvm.org/docs/LangRef.html#alloca-instruction)
instructions to allocate local variables. `alloca` takes an argument to
determine how many elements must be allocated. If we keep track of the
sizes of local allocations we can infer that the above code is safe if and only if:

```C
0 <= y < x //we'll assume x > 0 here
```

In our implementation, we simply keep around a map from allocaitons to their sizes.
Additionally, we track heap allocations by scanning for function calls to `malloc`.
This allows us to calculate maximum pointer index values for GEP instructions
where the types are unsized. Unfortunately, this is rather imprecise since
it tracks *exact* value dependencies and doesn't keep track of other ways
a pointer may be passed to a GEP. For instance, spilling a value to memory
and then re-loading it will cause our analysis to lose track of the original
allocation.

We had hoped to use LLVM's alias analysis or copy propagation tools
to increase the precision of allocation tracking. However, we couldn't get
these to work; they were difficult to integrate and didn't seem to track
pointer value propagation as we expected them to.

Furthermore, `bitcasting` operations complicate this process
even more, since they cause the "sizes" of memory allocations to
be interpreted differently.

```C
%1 = alloca i32, 10
%2 = bitcast i32* %1 to i8*
```

In the above code, a GEP that uses `%1` can safely index into elements 0 to 9.
However, a GEP that uses `%2` as the base can safely index into elements 0 to 31.
Since 4 `i8` values fit into one `i32` the allocation of `%2` represents a totally
different number of elements than `%1` even though they represent the result of the
same allocaiton operation. To avoid reasoning about the sizes of various types,
we did not implement this logic at another cost to precision. Any GEP instruction
using `%2` will not be instrumented by our code.

### Evaluation: Precision And Overhead


-----------

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

