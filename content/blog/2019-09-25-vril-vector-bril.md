+++
title = "Vril! Vector Bril"
extra.author = "Helena Caminal, Sachille Atapattu and Tuan Ta"
extra.author_link = ""
extra.bio = """
"""
+++

Why Vectors are cool to have!
---------------------------

## High-level idea
In principle vectors are just one-dimensional arrays. An array is a collection of elements that can be accessed by index. However, there are lot of applications where you would perform the same operation on each and every element of an array. This is what is called data-level parallelism (DLP). Some architectures are able to exploit this DLP by encapsulating several operations to different elements of one array into one vector operation. By doing that, they release the pressure in the front-end of the processor because they fetch and decode only once and then they can execute many identical operations concurrently. One of the challenges is for the workloads to exhibit sufficient data-level parallelism to achieve full utilization.

There are three popular options to program for vector architectres, a.k.a. *vectorization*:
- manual vectorization: Explicitly using vector operations in assembly or intrinsics.
- user annotations or [pragmas](https://info.ornl.gov/sites/publications/files/Pub69214.pdf): Help the compiler find the *vectorizable* regions of code and inform the compiler about lack of dependences and other situations that would typically restrict the vectorization.
- auto-vectorization: Rely directly on the compiler vectorizer which is typically enabled by the flag `-ftree-vectorize` and by default at `-O3` in [gcc](https://www.gnu.org/software/gcc/projects/tree-ssa/vectorization.html).
Programmer lead vectorization allows explicit control, but usually becomes architecture specific and assumes vector-like programming from the programmer.

Auto-vectorization aims to improve the programmer's productivity by performing a compiler pass to automatically generate vector instructions in a given program. However, in practice help from the programmer is needed to achieve competitive vectorized codes, either fully manual or with the user annotations.

An additional benefit of auto-vectorization and most user annotations is the decoupling the program specification and underlying execution which is architecture-specific.The vectorizable source will be portable to different architectures and the compiler will dictate if it can be vectorized or how well it can exploit the program's DLP in order to generate the architecture-specific vector instructions.

This project is aiming at having these general vector specifications at the compiler-level which:
- permits intrinsics or vector assembly instructions to be translated into bril IR
- permits architecture specific backend to easily generate an executable
- naturally expose opportunities for vector optimizations at bril
- offer to the compiler (automatic) vectorizer the possibility of generating bril IR vector operations that can later be mapped into ISA vector instructions

## Link to project
[Vril](https://github.com/sa2257/vril) is our public repository.

Arrays
------------------

## Arrays in Vril
Definition of an array in Vril is a sequence of elements of any type (no type checking carried out) and the length being any Bril literal. 

## Initializing arrays in Vril
Vril extends the grammar of Bril to include arrays by adding an `init` operator to initialize an array of length `l` as follows,
```
array_name : array = init l
```

Vril adds array operations to Bril of the form,
```
array/ variable: type = aop args
```
## Supported array operations
Vril can operate on arrays in two ways: 1) We can perform *scalar* accesses to arrays by moving one element of the array to a variable and performing any arithmetic or logic operation that are already supported in the original bril, and 2) we can perform *vector* accesses to the arrays by using new vector operations that take arrays as arguments directly.

In order to move data from an element of an array `arr[idx]` to a variable `var`, we can use `a2v` op:
`var: int = a2v arr idx`
Similarly, we can move a value stored in a variable `var` into one of the array's element `arr[idx]`:
`array_name: int v2a var index`

To put a constant into an array, the constant should first be put to a variable.
To operate with a scalar operator (for instance `print`) on a single element, the array element has to first be put to a variable.

Vector ops
----------------------------

## Vector ops in Vril
There are two types of vector ops in Vril: configuration ops and arithmetic/logic ops. Vector arithmetic or logic operations take arrays as arguments and produce an array or a scalar value as a result. Instead, configuration ops help in configuring a vector state.

## Adding vector ops to Vril
For now there is only one configuration op: `setvl`. 
`vl: int setvl val`
is used to request `val` lanes to the IR and `setvl` returns `vl:=val` if `vl<maxvl` or `vl:=maxvl` if `vl>=maxvl`.
We have also implemented only one arithmetic vector op `vadd` as a proof of concept:
`arr3: array vadd arr1 arr2'
Which takes two arrays as arguments and performs element-wise vector addition, and stores it into a third array.

## Simulating vector ops in Vrili (Vril interpreter)
We have extended Brili to interpret vector operations.

Evaluation
----------------------------

## Claim

## Comparison

## Benchmark

## Data Analysis

## Metrics

## Experimental Design

## Presentation

Conclusion
--------------------------------------
We have extended bril to support array types. We have added two new operations to move data in and out of the arrays so that we can emulate data movements between an array and what it would be a scalar register. We have also extended bril to support vector operations of two types: configuration and arithmetic. Configuration operations allow to modify a vector state and arithmetic operations perform operations on array arguments.

The goal of this exercise was to understand how much an IR needs to change in order to express data-parallel operations. For that, we extended bril to be able to express vector operations and to compare their potential against a traditional scalar set of operations. For that we written a benchmark for vector-vector add (vvadd), which adds the elements of two arrays and stores their results into a third array in two versions: a scalar code and a vector code. The end goal is to verify that the CFG generated is very similar for both codes, it should contain the same number of basic blocks (BB). However, the vector code hops on the BB involved in the `for-loop` statement *vector length* times less than the scalar code. 


[cs6120]: @/_index.md
