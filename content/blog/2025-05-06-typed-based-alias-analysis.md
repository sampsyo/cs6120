+++
title = "Typed Based Alias Analysis"
[extra]
latex = false
bio = """
Michael Xing is a Masters of Science student studying CS at Cornell and a Software Engineer at Microsoft. When not writing code, he enjoys travelling the world, riding roller coasters, and playing the piano. And video games.

Gerardo Teruel is a Masters of Engineering student at Cornell and used to be a Software Engineer at Etsy and Wizeline. When not thinking about software, he enjoys cooking, biking, and hiking with his dog Cardamom.

Arnav Muthiayen is an undergraduate student studying ECE and CS at Cornell. In his free time, he enjoys playing basketball and watching films.
"""
[[extra.authors]]
name = "Michael Xing"
[[extra.authors]]
name = "Gerardo Teruel"
[[extra.authors]]
name = "Arnav Muthiayen"
+++

One of the largest challenges facing an optimizing compiler is the fact that multiple variables in a program may refer to the same underlying memory, known as aliasing. The possibility of aliasing results in many compiler optimizations becoming unsafe, resulting in the compiler being unable to apply many seemingly reasonable optimizations unless it is able to conclusively determine that two variables are in fact not aliases.
This week’s paper, “Type-Based Alias Analysis” by Amer Diwan, Kathryn S. McKinley, and J. Eliot B. Moss, proposes a method of using type information and other statically available information like field names in order to better find variable pairs that are guaranteed not to be aliases.


## Background

Many reasonable compiler optimizations require that the variables they operate upon do not alias the same piece of memory. For example, the following code:
```C
bool myfn(int* a, int* b) {
	*a = 1;
	*b = 2;
	return *a == *b;
}
```
It would be convenient if the compiler could optimize the return value to be the constant false. Especially if a and b were dead after this operation, the entire function could be inlined to simply be false. However, if the pointers a and b both refer to the same memory location, then the result would actually be true, since \*a and \*b would fetch the same value.
For this reason, compiler optimizations depend upon alias analysis, where the compiler attempts to statically determine which variables could alias which others. Specifically, the compiler is generally looking for cases where two variables must not be aliases, as those are where opportunities for optimization typically arise.
A naive approach to alias analysis could involve a simple dataflow analysis, checking for what possible memory locations a variable could point to. However, due to the conservative nature of this analysis, it necessarily results in a large number of false positives.

## Summary

Diwan, McKinley, and Moss propose three new, progressively more sophisticated methods of performing alias analysis on statically typed programming languages.
First, the authors use type information to compare two fields. If the types of two fields are not the same, and they also have no subtypes in common, then by the semantics of most languages, they cannot alias the same memory (at least, not without triggering undefined behavior).
Second, the authors extend the analysis with field names. Even for two objects of the same type, accessing two different fields on two different objects cannot be aliases of the same memory.
Third, the authors extend the analysis again by also performing a (flow-insensitive) analysis of assignments, to check if a common subtype is ever assigned to variables of two different parent types. If not, then any variables of those two parent types cannot be aliases either, despite sharing a subtype.
The authors show that their later analyses are within 2.5% of a theoretically perfect analysis when applied to Redundant Load Elimination. One of the main contributions they emphasize is how they also performed a dynamic and limit analysis, in addition to the static analysis.

## Annotations, Rich Types, and Alias Analysis


One of the aspects of alias analysis is that it underestimates the true count of aliasing in a program. In TBAA it might be the case that we write a function that may alias but that in the total usages of the function it never actually aliases. In this scenario, we lose the opportunity to optimize some code, but we accept this tradeoff in order to keep the analysis fast and correct. In particular, TBAA is really powerful as it only relies on language constructs to perform the analysis so it doesn’t require special modifications to the source code to improve the results of the alias analysis. But this same approach opens more doors for optimizations if the source language has a richer type system or if there is a way for programmers to provide hints to the compiler. An example is the restrict keyword in C, where the programmers inform the compiler there will be no aliasing between parameters: 
```C
//source code
bool myfn(int* restrict a, int* restrict b) {
	*a = 1;
	*b = 2;
	return *a == *b;
}

// optimized version
bool myfn(int* restrict a, int* restrict b) {
	*a = 1;
	*b = 2;
	return false;
}
```

In languages with rich type systems the type of the parameter might provide enough information to the compiler so that annotations are not necessary. For example, if we define a function in C++ that receives [unique pointers](https://en.cppreference.com/w/cpp/memory/unique_ptr), we can achieve the same optimization without the need for the use of a restrict keyword: 

```C++
//source code
bool myfn(std::unique_ptr<int> a, std::unique_ptr<int> b) {
	*a = 1;
	*b = 2;
	return *a == *b;
}

// optimized version
bool myfn(std::unique_ptr<int> a, std::unique_ptr<int> b) {
	return false;
}
```
In fact, we were even able to optimize assignments to the pointer since we know that they will not be used outside of the scope of the function. The same idea can be used in languages that support immutable types, like [records in Java](https://openjdk.org/jeps/395). 


```Java
//source code
record MyInt(int value) {}
boolean myfn(MyInt a, MyInt b) {
	MyInt a1 = new Record(1)
 	MyInt b1 = new Record(2)
	return a1.value = b1.value
}
// optimized version
boolean myfn(MyInt a, MyInt b) {
	return false;
}
```

So although the original paper did not consider annotations or richer type constructs, the idea can easily be extended to provide even better results for alias analysis. 

## Measuring an Upper Bound

Despite the large number of compiler optimization techniques, it is often difficult to gauge how impactful an optimization will be in the performance of an arbitrary program; the most common approach is to use a set of benchmarks execute them a number of times in the original source  and in the optimized source do a pairwise comparison (usually in ratios) and aggregate the gains using the arithmetic or harmonic mean. This approach is not perfect, but makes it easy to compare different optimizations and is standard in literature. Diwan, McKinley and Moss used a completely different approach for TBAA, one that relies on establishing an upper bound.

The idea is to measure an upper bound: what is the maximum number of instructions that can be removed from a source program and measure how many of them were removed in the optimized source with different versions of alias analysis. With this approach in place, the authors are not only able to claim that x% of instructions were eliminated, but also that the alias analysis that is presented is within 2.5% of a perfect alias analysis algorithm! Their evaluation and methods are one of the key strengths of the paper and provide compelling evidence not just about the performance improvement, but also about how precise the analysis is.

## Connections to Computing Landscape

As compilers have evolved, TBAA has been applied to not only type safe languages but also unsafe languages like C and C++. The introduction of the “strict aliasing rule” in the [C99 standard](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1256.pdf) allowed compilers to safely assume pointers of incompatible types wouldn’t alias, treating any violations as undefined behavior. There are some exceptions, though, such as character pointers. Compilers such as GCC and LLVM exploit this rule to enable more aggressive optimization. GCC enables [-fstrict-aliasing](https://gcc.gnu.org/onlinedocs/gcc-7.5.0/gnat_ugn/Optimization-and-Strict-Aliasing.html) by default at higher optimization levels like -O2 and LLVM/Clang enables a similar [TBAA metadata system](https://llvm.org/docs/LangRef.html#tbaa-metadata) by default at all optimization levels. A notable drawback is that legacy code that violated these strict aliasing rules could break unexpectedly, prompting the creation and usage of the [-fno-strict-aliasing](https://gcc.gnu.org/onlinedocs/gcc-7.5.0/gnat_ugn/Optimization-and-Strict-Aliasing.html) flag to preserve the previous behavior; moreover, Clang provides [TypeSanitizer](https://clang.llvm.org/docs/TypeSanitizer.html) (enabled with -fsanitize=type flag), a run-time library that uses TBAA metadata and dynamic instrumentation to detect strict type aliasing violations.

TBAA became popular in the first place because it offered a fast, conservative way to disambiguate pointers, while staying simple enough to work across separately compiled modules. It built on earlier techniques like [Steensgaard’s Points-to analysis](https://dl.acm.org/doi/10.1145/237721.237727), but emphasized practicality in its speed over full precision. That balance helped it gain traction in real-world compilers.

Under the hood, compilers like LLVM have their own strategies for using TBAA in practice. Because LLVM IR itself doesn’t track memory types, language frontends (like Clang for C/C++) explicitly add [metadata annotations](https://llvm.org/docs/LangRef.html#tbaa-metadata) that reflect their respective language-specific type hierarchies. While commonly used to implement C/C++ strict aliasing rules, this flexible metadata system can similarly represent custom aliasing rules and type relationships for other languages. LLVM’s optimization passes then use this metadata to perform transformations such as load/store reordering and common subexpression elimination, knowing that pointers with incompatible types won’t alias.
