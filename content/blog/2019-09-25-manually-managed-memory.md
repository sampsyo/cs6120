+++
title = "Manually Managed Memory in Bril"
extra.author = "Drew Zagieboylo & Ryan Doenges"
extra.bio = """
  [Drew Zagieboylo](https://www.cs.cornell.edu/~dzag/) is a 3rd year PhD student researching Security, HW design and Programming Languages. He enjoys rock climbing and gaming in his free time.
  [Ryan Doenges](http://ryandoeng.es/) is a 3rd year PhD student studying programming languages for networking.
"""
+++

### Pointers and Heap Allocated Memory

Our goal was to add *pointer types* to Bril. _Pointers_ represent references to
manually managed read/write memory cells which can persist outside of function
scope. Furthermore we support C-style arrays such that pointer arithmetic
instructions can be used to index into allocated memory regions. Lastly, we
wished to ensure that value typechecking was still supportable for our new
instructions (however we did not implement a typechecker). Our pointer types are
meant only for value checking (i.e. every pointer type totally specifies the
type of its contents); they do not include bounds or alias information to
prevent memory safety bugs.

### Design Overview

We added manually managed memory and typed pointers to Bril while keeping the
layout of data hidden from Bril programs. The API for working with these
heap-allocated pointers was inspired by [LLVM's manually allocated stack pointer
API](https://llvm.org/docs/LangRef.html#memory-access-and-addressing-operations).
Pointer types include the data type to which they refer: no `void*` magic
allowed. Furthermore, the representation of data in memory is abstract. In
a type system that just consists of integers, booleans, and pointers this is not
too strong a statement, but something like LLVM's `getelementptr` would allow
structs to be added while still hiding type sizes from Bril programs.
This depends on the fact that, unlike in C, you cannot do bytewise arithmetic on
Bril pointers to determine the size of things in memory or extract the value of
a pointer as an integer address.

This design leaves the Bril interpreter/compiler complete freedom to choose how
data including pointers are represented and allocated; this separates the high
level computaional usage of pointers and the low level (and likely
platform-dependent) implementation details while still providing usable manually
managed memory to programs.

#### Pointer Syntax & Representation

We expanded the Bril type syntax with ```ptr<TYPE>```, which denotes a pointer
to a value in memory of type `TYPE`.
There is no additional syntax for pointer values, since pointer representation
is abstract: the only way to produce something of type `ptr<T>` is by using the
language's memory allocator. There is no address-of operator (like C's `&`)
either.

#### Allocating Memory

We added a typed `alloc` primitive to Bril.
Bril's `alloc` works like C's `malloc`, but the argument passed to `alloc`
represents the number of elements to allocate, rather than the number of bytes
to allocate.

In Bril, allocating a pointer to 10 ints looks like this:
```C
 ten: int = const 10;
 myptr: ptr<int> = alloc ten;
```

Doing the same in C would require invoking the `sizeof` operator to determine
how much space an `int` takes up in memory, but that lets the program know
something about the representation of data in memory. Bril's element-size
allocator avoids this.

#### Modifying Memory

Like in assembly languages, Bril pointers can be used to access memory through
`load` and `store` instructions. These operations take a pointer as their first
argument and work like pointer dereferencing in C. *Loads* correspond to
read operations and *stores* correspond to writes. As an example, both of the
following programs will print the value `4`.

C Implementation:
```C
int* myptr = malloc(sizeof(int) * 10);
*myptr = 4;
printf("%d\n", *myptr)
```

Bril Implementation:
```C
 ten: int = const 10;
 four: int = const 4;
 myptr: ptr<int> = alloc ten;
 store myptr four;
 v: int = load myptr;
 print v;
```

#### Pointer Arithmetic
So far we can only use loads and stores to access the first cell in our
allocated memory region, since that's where the pointers returned by `alloc`
point to.

To index into the memory region, Bril programs can use our typed `ptradd`
instruction, which allows a program to add an integer to a pointer to produce
a new pointer.

The code snippets below access the second element of some already allocated
memory region.

C Implementation:
```C
int* myptr;
// ... allocate memory ...
printf("%d\n", myptr[1]); // myptr[1] === *(myptr + sizeof(int)*1)
```

Bril Implementation:
```C
one: int = const 1
myptr_1: ptr<int> = ptradd myptr one
v: int = load my_ptr1
print v
```

#### Deallocating Memory

In general, `free` in Bril works exactly the same as it does in C. You can use
any reference to the same allocation to free it; however, double frees or
free-ing a pointer which doesn't refer to the beginning of an allocation are
illegal. That means the following programs both result in bad behavior at
runtime:

Error Program 1:
```C
ten: int = const 10;
myptr: ptr<int> = alloc ten;
free myptr;
free myptr;
```

Error Program 2:
```C
ten: int = const 10;
myptr: ptr<int> = alloc ten;
myptr_10: ptr<int> = ptradd myptr ten
free myptr_10;
```
Furthermore, (also like C) Bril does not prevent memory leaks by default. In
other words, programs may `alloc` memory that they never `free`.

For a larger example of how pointers can be used in Bril, the following C code:
```C
int* vals = malloc(sizeof(int)*10);
vals[0] = 0;
for (int i = 1; i < 10; i++) {
  vals[i] = vals[i-1] + 4;
}
```

Would be roughly equivalent to the following Bril code:
```C
 ten: int = const 10;
 zero: int = const 0;
 one: int = const 1;
 neg_one: int = const -1;
 four: int = const 4;
 vals: ptr<int> = alloc ten
 store vals zero;
 i: int = const 1;
 i_minus_one: int = add i neg_one;
loop:
 cond: lt i ten;
 br cond done body;
body:
 vals_i: ptr<int> = ptradd vals i;
 vals_i_minus_one: ptr<int> = ptradd vals i_minus_one;
 tmp: int = load vals_i_minus_one;
 tmp: int = add tmp four;
 store vals_i tmp;
 i = add i one;
 i_minus_one = add i_minus_one one;
 jmp loop;
done:
 free vals;
 ret;
```

### Implementation

We implemented our design by extending the Bril interpreter with for pointers,
heap memory, and runtime error checking.

#### Pointer Representation

A Bril pointer is represented by a pointer with two fields: a *Key* that points
into the [*Heap*](#heap-memory), and a *tag* that tells the runtime which kind
of data the memory cell should be used for. Type tags can be `"int"`, `"bool"`
or `"ptr"`. Type tags are checked whenever a Bril program does a `store` to make
sure that the cells of the allocated memory were allocated to store something of
that type. All pointers, regardless of what they point to, have the same type
tag. This is because all pointers have the same in-memory representation, so
ensuring a cell for pointers is only ever used for pointers can be done without
worrying about what type of data the pointer points to.

#### Heap Memory

Memory itself is represented by the `Heap` data structure. It implements all the
operations exposed to Bril programs. Concretely, the `Heap` implementation is
a Javascript `Map` that maps `number` keys to arrays of objects. Its `alloc`
method extends the map to map a fresh key `number` to a new array of the
requested size and returns an opaque *Key* wrapping the `number`.

A *Key* is a pair of a `base` and an `offset`. The `base` is the key used to
look up the array in the heap `Map` and the `offset` is an index into the array.
Freshly allocated pointers have `offset == 0`, and pointer arithmetic can only
change the `offset`. In this way, we keep track of which allocation any given
pointer belongs to, regardless of "where" it may point in memory. Notably this
implementation of a Heap does not model "a single contiguous memory space"; each
allocation represents a continguous space and allocations are otherwise
unrelated.

The `free` method deletes entries from the internal map, so we are relying on
the base Typescript Map implementation and the Javascript runtime garbage
collection to actually free physical memory dynamically. We don't implement any
interesting memory allocation strategies based on physical memory layout and
simply let the runtime do the work. While smart, type-aware memory allocators
are an interesting area of performance optimization, we felt that rabbit hole
was outside the scope of this small project.

### Notable Challenges

In an interpreter setting, where we can rely on another language runtime to do
the real physical memory allocation and garbage collection for us, this is not
a terribly complex addition to Bril. However, there were a few small details
that were tricky to get exactly right.

Firstly, we had to handle how to parse and represent new types that are
parameterized on other types. We wanted to syntactically enforce, with the
parser, that pointer types had to be fully specified.

The original type parsing specification looked like this:

```C
type: CNAME
```

And created an AST node where the type was specified as a `string`.

Our new version looks like this:
```C
type: "ptr<" ptrtype ">" | basetype
ptrtype: type
basetype: CNAME
```

And we create an AST node where the type is specified as a (potentially nested)
object with one field named "ptr". For example, the AST representation of a node
with type `ptr<bool>` looks like

```C
type: { ptr: "bool" }
```

We could have decided to maintian the pointer type abstract representation as
a string, but that would require re-parsing that type string repeatedly
throughout the interpreter. While we avoided that problem, we now had to deal
with the annoyance that some types were represented as strings and others were
JSON objects. This lead to a fair bit of refactoring and slightly tricky runtime
typechecking, but we decided it was worth it compared to having to do
sophisticated string pattern matching at runtime.

### Evaluation

We evaluated our implementation through qualitative testing rather than
quantitative measurement. We wanted to evaluate the correctness of our code and
see if we threw reasonable errors under all erroneous conditions. We created
a number of test cases that stress pointer arithmetic (similar to the large
example presented earlier in this post). Key features to test were:

 - Allocating memory of various sizes
 - Reading & writing memory
 - Re-writing memory
 - Ensuring that pointers to pointers function correctly

Additionally, we needed to check a number of "bad" cases, which we expected the
interpreter to catch and report as errors with reasonable error messages:

 - Passing non-pointers to `load`, `store` or `free` operations
 - Allocating pointers with non-positive size
 - Failing to free memory by the end of the program
 - Freeing the same allocation multiple times
 - Trying to free a pointer into the middle of an allocation region
 - Accessing memory "out of bounds" of a given access
 - Writing the wrong type of data into a pointer (e.g. store `int` into
   a `ptr<bool>`)
   - N.B. that "reading" data of the wrong type is still allowed, which actually
     mirrors the current interpreter implementation for other operations
