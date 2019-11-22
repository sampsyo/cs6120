+++
title = "Finding and Understanding Bugs in C Compilers"
extra.author = "Christopher Roman, Zhijing Li"
extra.bio = """
[Zhijing Li](https://tissue3.github.io/) is a 2nd year Phd student at ECE department, Cornell. She is interested in compiler optimization and domain-specific languages. \n
"""
extra.latex = true

+++







## Safety Mechanisms

A program can crash for various reasons. It might be because of compiler failures, but more often, due to unsafe programs. Csmith wants to avoid these pitfalls with proper safety mechanisms.

**Discussion:** Think about what safety issues can be caused by the randomly generated programs.

A summary of all mechanisms shown below. We will provide detailed explanations for it.

<img src="safety_mechanism.jpg" style="width: 70%">

### Integer Safety

The safety problem of integers comes from undefined behaviors (UB) such as signed overflow:

```c++
int signedOverflow(int x) {
    return x+1 > x; // either true or UB due to signed overflow
}
```

and shift-past-bitwidth:

```c++
int shiftPastBitwidth() {
    return 1 << 32; // UB when evaluated on 32 bit platform 
}
```

Csmith generates wrapper functions for arithmetic operators whose operands might overflow according to the compiler standard, though there are tricky UBs not defined and the authors have to figure out themselves (such as `INT_MIN % -1`).

### Type Safety

The tricky aspect of C's type safety is qualifier safety. Modifying constant-qualified or volatile-qualified objects through nonconstant references is [undefined behavior](<https://wiki.sei.cmu.edu/confluence/display/c/EXP40-C.+Do+not+modify+constant+objects>).

```c++
const int **ipp; // the value pointer to shall not be modified
int *ip;
const int i = 42;
 
void constViolation(void) {
  ipp = &ip; // UB
  *ipp = &i; // Valid
  *ip = 0;   // Modifies constant i (was 42)
}
```

Csmith ensures type safety with type checks.

### Pointer safety

The first kind of pointer safety problem is null pointer dereference.

```c++
int a = 10;
void nullDereference(int *p){
    *pi = a; // cause execption if p is NULL
}
```

 This can be easily avoided by dynamic checks.

```c++
int a = 10;
void safeDereference(int *p){
    if( p != NULL){
    	*p = a; 
    }
}
```

However, there is no reliable method to identify an invalid pointer that points to a function-scoped variable.

```c++
void invalidDereference(int *p){
    int a = 10;
    if( p != NULL){
    	*pi = a; 
    }
}
// outside this function, we cannot dereference or compare p with other pointer 
// before it becomes valid again!
```

**Question:** What could be the solution here? 

One way is to force a pointer, which points to a function-scoped variable, to never outlive the function; however, this is too restrictive. Csmith instead chooses to do pointer analysis that is flow sensitive, field sensitive, context sensitive, path insensitive, and array-element insensitive. 

### Effect Safety

In the C99 standard, undefined behavior can be caused by “the order of evaluation of the function designator, the actual arguments, and subexpressions within the actual arguments", and if "between two sequence points, an object is modified more than once, or is modified and the prior value is read other than to determine the value to be stored."

```c++
void undefinedExcutionSequence(){
	some_func(printf("first\n"), printf("second\n"));; // which printf is called first?
	i = ++i + 1; // what is i?
}
```

Csmith conservatively analyzes the *effect* of every expression. An effect contains *may-read* and *may-write* locations of the expression. The same location cannot be both read and write or written twice except for assignment. This is done incrementally. For newly generated code, a check is performed to decide abandon or keep the code. For example, when generating  `p + funcMayModifyP()`, Csmith will abandon `funcMayModifyP()` .

### Array Safety

Out of boundary safety issues are always there for arrays.

```c++
void outOfBound(){
    int a[2];
    a[2] = 10; // out of boundary
}
```

The example I give is simple to avoid, but when the index of `a` is a variable, it is hard to tell whether it is in-bounds or not. Csmith only generates index variables of `for` loops counter and ensures the `for` loop never exceeds the boundary. For arbitrary index variables, Csmith applies the modulo operator. If both techniques do not work—for example, when array length increases—Csmith emits explicit checks against array lengths.

### Initializer Safety

Csmith initializes variables right after declaration and bans `goto` statement to ensure the execution is in order.

**Question:** What kind of cases can be omitted due to the current design of program generation and safety constraints? 

### Global Safety

Because Csmith generates the code incrementally, newly generated code can threaten the old code.

```c++
incrementallyGeneratedUnsafeProgram(){}
    int *p = &i;
    while (...) {
        *p = 3; 
        p = 0; // unsafe because of the back-edge
    }
}
```

Csmith performs checking at each newly generated line except for loops. Loops are checked at the end when the back-edge is logically created. If unsafe lines appear, Csmith deletes line by line until the safety requirements are satisfied.

## Design Tradeoff

### Allow Implementation-defined Behavior

Implementation-defined behavior is equivalent to [unspecified behavior](<https://en.wikipedia.org/wiki/Unspecified_behavior>), which may vary on different implementations of a programming language. Csmith designers believe it is unrealistic to "retain a single interpretation across all possible choices of implementation-defined behaviors". They allow Csmith to give different outputs when compilers has implementation-defined behavior at 

1. the widths and representations of integers
2. behavior when casting to a signed integer type when the value cannot be represented in an object of the target type
3. results of bitwise operations on signed integers. 

There are roughly three kinds of compiler targets:

| Targets               | `int` bit length | `long` bit length |
| --------------------- | ---------------- | ----------------- |
| x86-64                | 32               | 64                |
| x86, ARM, and PowerPC | 32               | 32                |
| MSP430 and AVR        | 16               | 16                |

Csmith performs testing within but not cross the equivalent classes.

### No Ground Truth

Csmith does not have a ground truth since it is unrealistic to have a human check each program. Instead, it takes the majority vote. In practice, two uncorrelated compilers have not output the same incorrect output, according to the authors. The explanation to this is the substantial diversity among intermediate languages (IRs).

### No Guarantee of Termination

A Csmith-generated program can be of any length. In practice, a time-out function is called to terminate program that takes too long to finish.

### Target Middle-End bugs

Csmith targets checking how compilers perform transformations on IRs rather than standard conformance as commercial test suites. For instance, Csmith does not spend efforts to test how compilers handle long identifier names.

There are several choices made because of the target:

1. The Csmith designers manually tune the 80 probability variables to generate programs with a balanced mix of arithmetic and bitwise operations, of loops and straight-line code, and of single-level and multi-level indirections, *etc*.
2. Encouraging Csmith to generate idiomatic code, e.g. loops that access all elements of an array.
3. Discouraging Csmith from generating source-level diversity that is unlikely to improve the “coverage” of a compiler’s IR, e.g. additional levels of parentheses around expressions.
4. Designing Csmith efficiently generates runnable programs of a few tens of thousands of lines long in a few seconds.

## Evaluation

### Opportunistic Bug Finding

The Csmith designers tested 11 compilers and reports the errors to the developers. The commercial compiler developers do not care, while GCC and LLVM team response quickly. By the time the paper is formalized, 79 GCC bugs and 202 LLVM bugs (2% of all LLVM bug reports) are reported and most of them are fixed. CompCert is such a good compiler that the under-development version of CompCert is the only compiler Csmith cannot find wrong-code errors after 6 CPU-years of testing.

### Bug Types

Before we move on, I would like to introduce bug types for understanding the experiment results:

1. A *crash error* is one that crashes the compiler during compilation.
2. A *wrong error* is one that crashes the program during run-time.
3. A *slient wrong error* is a wrong error that does not produce a compiler warning during compilation.
4. An *assertion failure* is an LLVM internal consistency check failure.
5. An *internal compiler failure* is a GCC internal consistency check failure.

### Quantitative Comparison of GCC and LLVM Versions

The following figure shows the compilation and execution results of LLVM 1.9–2.8, GCC 3.[0–4].0, and GCC 4.[0–5].0 given the input of 1,000,000 Csmith randomly generated programs. Every program was compiled at –O0, –O1, –O2, –Os, and –O3. A test case is valid if every compiler terminated within five minutes and if every compiled random program terminated within 5 seconds. The top and bottom row are the error rates of different compiler versions. The authors also find the source of the compilation error and plot it in the middle row.

<img src="llvm_gcc.png" style="width: 100%">

### Bug-Finding Performance as a Function of Test-Case Size

The goal of designing Csmith is to find many defects quickly, and to what size the program should Csmith generate to achieve that goal becomes a question. When reporting the error, we preferred smaller programs over larger one because they are easier to debug and report. The figure below shows the experiment performed to learn the error number and runtime tradeoff given the same run-time. 

<img src="size_error.jpg" style="width: 60%">

### Bug-Finding Performance Compared to Other Tools

Finally, there is also a performance test of Csmith against other bug finding tools. Given the same time of bug finding, Csmith is much more efficient than existing testing tools.

<img src="other-tool.jpg" style="width: 60%">

### Code Coverage

Finally, there is also a code coverage test on Csmith generated programs as shown in the following table. 

<img src="coverage.jpg" style="width: 70%">

Adding 10,000 Csmith generated program to the existing test suite of LLVM and GCC plus 10,000 Csmith does not to increase the coverage a lot. 

**Question:** 

1. The authors do not come up with a good explanation to the code coverage issue. What might be the reason? 
2. Tests that are randomly generated will never be like tests that are created by humans. For example, the factorial function will almost never be randomly generated. Does this mean that this kind of testing is still useful? Why and how?
