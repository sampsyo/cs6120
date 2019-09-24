+++
title = "Manually Managed Memory in Bril"
extra.author = "Drew Zagieboylo & Ryan Doenges"
extra.bio = """
  [Drew Zagieboylo](https://www.cs.cornell.edu/~dzag/) is a 3rd year PhD student researching Security, HW design and Programming Languages. He enjoys rock climbing and gaming in his free time.
  [Ryan Doenges](http://ryandoeng.es/) is a 3rd year PhD student...
"""
+++

### Pointers and Heap Allocated Memory

Our goal was to add *pointer types* to Bril. _Pointers_ represent references to manually managed read/write memory cells which can persist outside of function scope. Furthermore we support C-style arrays such that pointer arithmetic instructions can be used to index into allocated memory regions. Lastly, we wished to ensure that value typechecking was still supportable for our new instructions (however we did not implement a typechecker). Our pointer types are meant only for value checking (i.e. every pointer type totally specifies the type of its contents); they do not include bounds or alias information to prevent memory safety bugs.

- What did you do? (Include both the design and the implementation.)

### Design Overview

We designed Bril pointers to be very similar to [LLVM's manually allocated stack pointers](https://llvm.org/docs/LangRef.html#memory-access-and-addressing-operations), except that Bril pointers refer to heap-allocated memory and offer much more restricted operations. Namely, this means that pointer types completely define the data type to which they refer and and pointer representations are explicitly abstract. This means that, unlike in C, you cannot use a Bril pointer as an argument to an `add` operation; even if we did support casting it would be meaningless to "add" a Bril pointer and a Bril integer. More generally, the Bril interpreter/compiler has complete freedom to choose how pointers are represented and allocated; this separates the high level computaional usage of pointers and the low level (and likely platform-dependent) implementation details.

#### Pointer Syntax & Representation

First we expanded the Bril syntax to support pointer types of the form ```ptr<TYPE>```. 
Since we want Bril types to be fully statically determined, `TYPE` must itself be a complete and legal Bril type.
Therefore, the type ```ptr<ptr<bool>>``` is well-formed, while the type ```ptr<ptr>``` is not.
Pointer *values*, however, have no syntactic representation, since we don't actually want their representation to be concrete. This implies that `const` operations cannot ever produce a pointer type *and* that the compiler/interpreter has complete control over how pointers are implemented.

#### Memory Allocation

`alloc` works almost the same as C's `malloc`, except that the argument passed to `alloc` represents *the number of elements to allocate*. In C, `malloc` expects its allocation size argument to specify a number of *bytes*, which are only loosely tied to the type of data to which the pointer refers.

C Allocation of a Pointer to 10 ints:
```
int* myptr = malloc(sizeof(int) * 10);
...
```

Bril Allocation of a Pointer to 10 ints:
```
 ten: int = const 10;
 myptr: ptr<int> = alloc ten;
 ...
```

Note that, in the C code, we needed to explicitly use a platform-dependent `sizeof` operator to translate our `int` type into a number of bytes. In Bril, that is unnecessary since the compiler/interpreter can determine how many bytes to allocate based on its representation of integers.

#### Accessing Memory

Like assembly languages, pointers can be used to access memory through `load` and `store` instructions. These operations take a pointer as their first argument and are analogous to pointer dereferencing in C. *Loads* correspond to read operations and *stores* correspond to writes. As an example, both of the following programs will print the value `4`.

C Implementation:
```
int* myptr = malloc(sizeof(int) * 10);
*myptr = 4;
printf("%d\n", *myptr)
```

Bril Implementation:
```
 ten: int = const 10;
 four: int = const 4;
 myptr: ptr<int> = alloc ten;
 store myptr four;
 v: int = load myptr;
 print v;
```

With these operations, we can only access the first memory cell in our allocated memory region, since that's where pointers returned by `alloc` point to. In C, you can use arithmetic operations on pointers to get new pointers that reference other bytes in that memory region. We include a `ptradd` operation to support this kind of functionality, which allows a program to add an integer to a pointer to produce a new pointer. The code snippets below access the second element of some already allocated memory region:

C Implementation;
```
int* myptr;
...
printf("%d\n", myptr[1]); // myptr[1] === *(myptr + sizeof(int)*1)
```

Bril Impleentation:
```
...
one: int = const 1
myptr_1: ptr<int> = ptradd myptr one
v: int = load my_ptr1
print v
```

#### Deallocating Memory

In general, `free` in Bril works exactly the same as it does in C. You can use any reference to the same allocation to free it; however, double frees or free-ing a pointer which doesn't refer to the beginning of an allocation are illegal. That means the following programs both result in bad behavior at runtime:

Error Program 1:
```
ten: int = const 10;
myptr: ptr<int> = alloc ten;
free myptr;
free myptr;
```

Error Program 2:
```
ten: int = const 10;
myptr: ptr<int> = alloc ten;
myptr_10: ptr<int> = ptradd myptr ten
free myptr_10;
```
Furthermore, (also like C) Bril does not prevent memory leaks by default. In other words, programs may `alloc` memory that they never `free`.


For a larger example of how pointers can be used in Bril, the following C code:

```
int* vals = malloc(sizeof(int)*10);
vals[0] = 0;
for (int i = 1; i < 10; i++) {
  vals[i] = vals[i-1] + 4;
}
```

Would be roughly equivalent to the following Bril code:
```
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

To realize our design we had to complete the following tasks:
   - Define a concrete pointer representation
   - Implement a "memory allocator" that can create variable-size memory cells
   - Support runtime error detection via argument typechecking and initialization checking


#### Pointer Representation

In our interpreter implementation, pointers are objects with two fields: a *Key* (a datatype defined by the [*Heap*](#memory-allocator) implementation) that is used for actually looking up where data is stored; and a *tag* which is a string that tells the runtime which kind of data the memory cell should be used for. Unlike the surface syntax for Bril types, type tags must be one of the following values: `"int"`, `"bool"` or `"ptr"`.

We don't need separate tags for every possible pointer type because all pointers have the same representation (namely the struct we just defined). During `load` and `store` instructions, our interpreter uses the type tag to ensure that the memory cell to which the pointer refers actually holds (or can actually hold) data of the appropriate type.

#### Heap Memory

We added a new data structure to the Bril interpreter to store heap-allocated memory, which we unsurprisingly called a "Heap". This heap supports pretty much exactly the same set of operations as the Bril pointer instructions and does most of the heavy lifting for the interpreter, other than dynamic type checking. Our "Heap" is really just a `Map` that maps Typescript `numbers` to arrays of objects. `alloc` gets a fresh number (by incrementing a global counter, we don't do anything smart about re-using numbers) and puts an array of the requested size into the map with that number as its key. Lastly, `alloc` returns an opaque *Key* object that the interpreter can use for future *Heap* operations, such as `get`, `set` and `free`.

In reality, the *Key* is just another simple object that contains a `base` and an `offset`. Pointer arithmetic operations only modify the offset so we can keep track of whether or not any given pointer corresponds directly to an allocation (i.e. `offset == 0`). In this way, we keep track of which allocation any given pointer belongs to, regardless of "where" it may point in memory. Notably this implementation of a Heap does not model "a single contiguous memory space"; each allocation represents a continguous space and allocations are otherwise unrelated.

`free` deletes entries from the internal map, so we are relying on the base Typescript Map implementation and the Javascript runtime garbage collection to actually free physical memory dynamically. We don't implement any interesting memory allocation strategies based on physical memory layout and simply let the runtime do the work. While smart, type-aware memory allocators are an interesting area of performance optimization, we felt that rabbit hole was outside the scope of this small project.

#### An Alternative Heap Implementation

TODO Ryan put Array Buffer stuff here

### Notable Challenges

In an interpreter setting, where we can rely on another language runtime to do the real physical memory allocation and garbage collection for us, this is not a terribly complex addition to Bril. However, there were a few small details that were tricky to get exactly right.

Firstly, we had to handle how to parse and represent new types that are parameterized on other types. We wanted to syntactically enforce, with the parser, that pointer types had to be fully specified.

The original type parsing specification looked like this:

```
type: CNAME
```

And created an AST node where the type was specified as a `string`.

Our new version looks like this:
```
type: "ptr<" ptrtype ">" | basetype
ptrtype: type
basetype: CNAME
```

And we create an AST node where the type is specified as a (potentially nested) object with one field named "ptr".
For example, the AST representation of a node with type `ptr<bool>` looks like

```
type: { ptr: "bool" }
```

We could have decided to maintian the pointer type abstract representation as a string, but that would require re-parsing that type string repeatedly throughout the interpreter. While we avoided that problem, we now had to deal with the annoyance that some types were represented as strings and others were JSON objects. This lead to a fair bit of refactoring and slightly tricky runtime typechecking, but we decided it was worth it compared to having to do sophisticated string pattern matching at runtime.

### Evaluation

Primarily, our evaluation here was qualitative rather than quantitative; we simply want to ensure that our instructions operated correctly on correct inputs and throw reasonable errors under all erroneous conditions. We created a number of test cases that stress pointer arithmetic (similar to the one presented earlier in this blog). Key features to test were:

 - Allocating memory of various sizes
 - Reading & Writing memory
 - Re-writing memory
 - Ensuring that pointers to pointers function correctly

Additionally, we needed to check a number of "bad" cases, which the interpreter should catch as errors and print a reasonable error message:

 - Passing non-pointers to `load`, `store` or `free` operations
 - Allocating pointers with non-positive size
 - Failing to free memory by the end of the program
 - Freeing the same allocation multiple times
 - Trying to free a pointer into the middle of an allocation region
 - Accessing memory "out of bounds" of a given access
 - Writing the wrong type of data into a pointer (e.g. store `int` into a `ptr<bool>`)
 - Reading data into the wrong type of destination variable


##TODO empricial evaluation part