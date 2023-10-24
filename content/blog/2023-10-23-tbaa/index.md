+++
title = "How Type Systems Optimize Optimizers"
[extra]
authors = "Albert Xiao, Jan-Paul Ramos, Kei Imada, and Ryan Mao"
bio = """
    Kei Imada is a first-year Ph.D. student. He is interested in applying
    mathematical structures to type systems and formal verification to
    enable efficient development of parallel and distributed systems, and software
    defined networks.
"""
[[extra.authors]]
name = "Albert Xiao"
[[extra.authors]]
name = "Jan-Paul Ramos"
[[extra.authors]]
name = "Kei Imada"
link = "https://keikun555.github.io/"
[[extra.authors]]
name = "Ryan Mao"
+++
## Background


### What is alias analysis?

Let’s recall from the [pointer analysis lecture](https://www.cs.cornell.edu/courses/cs6120/2023fa/lesson/10/) what pointer analysis is.

For each pointer pair A and B, we determine whether A may-alias B, in other words, whether A and B could point to the same memory location.

The stricter problem of determining whether A would alias B is an intractable problem as covered in the lecture.

In the example below, the analysis of the code might tell us that A will not alias B, so we would be able to remove the star B equals B line, making the resulting program more efficient.

```c++
// a may-alias b?
bool foo(char *a, char *b) {
    *a = 'a';
    *b = 'b'; // remove if not
    return *a == 'a';
}
```


### Why do we want alias analysis?

We want to analyze aliases because many of our popular languages use pointers today, and some language-features require pointers, like polymorphism.

And oftentimes, these addresses point to memory allocated in the heap.

Loading memory from the heap is more random and less frequent than loading memory from the stack, which makes reading from the heap more prone to cache misses, decreasing the performance of the program.

Alias analysis lets us reason about which load instructions are necessary and which ones are not.


### Problems with alias analysis

The paper cites that before, alias analysis wasn’t used as much because (1) it was slow, (2) had a closed-world assumption, and (3) was only evaluated statically.


![alt_text](images/image1.png "image_tooltip")


Just static analysis didn’t tell compiler developers how effective the analyses would be in a real-world environment, in other words, applied to an optimization.

The closed-world assumption meant that the entire program was needed to do these analyses which meant that we couldn’t use them on compiled libraries, which meant we wouldn’t be getting modularity of code.

And slowness meant that the analyses were, well, slow.


### Type-based alias analysis

So that’s why Diwan, McKinley, and Moss thought of leveraging fast type systems of typed languages to enhance alias analysis.

Hence the name type-based alias analysis.


![alt_text](images/image2.png "image_tooltip")


![alt_text](images/image3.png "image_tooltip")


And they ended up with a near-optimal algorithm that is O(Instructions * Types).


![alt_text](images/image4.png "image_tooltip")


The main contributions of this paper come in three forms.



* Three implementations of type-based alias analysis, built on top of one another.
    * Type compatibility
    * Type compatibility + field names
    * Type compatibility + field names + flow-insensitive analysis
* Three different kinds of evaluations.
    * _Static evaluations_, or measures on the code itself
    * _Dynamic evaluations_ on an actual optimization, called redundant load elimination
    * _Limit analysis_ which compares against the quote-on-quote best case scenario.
* How well the analyses perform on an open world assumption.

There are two more background info we need to cover. Modula-3, the language they implemented this in, and a brief primer about redundant load elimination.


### Modula-3


### Redundant load elimination (RLE)


## Type-based alias analysis (TBAA)

Three type-based alias analysis (TBAA)

	Type compatibility

	Type compatibility + field names

	Type compatibility + field names + flow-insensitive analysis


## Evaluation

The authors use three different types of evaluation methods to examine their work.



* **Static Evaluation:** measure static measures of the analysis. In the paper, the authors examine the sizes of the may-alias sets returned by their algorithms, where smaller sizes correspond to more precise results. Crucially, the authors note that static evaluation alone may not provide a holistic picture of the performance of the analysis.
* **Dynamic Evaluation:** measure the actual runtime performance of the algorithm, after performing optimizations utilizing their analysis. The authors implemented RLE using the sets computed by their TBAA, and measured the wall clock runtime both with and without optimization. This evaluation, in conjunction with the static evaluation, provides a bit more insight into the results of the analysis.
* **Limit Analysis:** evaluates the gap between the analysis performance and a hypothetical ‘best’ analysis. The authors explore this gap using instrumentation in a runtime setting.


### Results: Static Evaluation

The authors run all three variants of TBAA (TypeDecl, FieldTypeDecl, and SMFieldTypeRefs) on their benchmarks and aggregate the sizes of may-alias local and global pairs within each benchmark. It’s worth noting that these three variants are each strictly stronger than the previous. In terms of the computed pair set size, there was a significant improvement between TypeDecl and the latter two analyses. Interestingly enough, there was not a very significant improvement between FieldTypeDecl and SMFieldTypeRefs. One possible explanation is that the field types inherently provide a ton of information, and bring the resulting analysis very close to a best-case analysis. 


### Results: Dynamic Evaluation

The authors performed RLE using the sets computed in their analysis and found improvements of 1% to 8% in wall clock runtime on the benchmarks, with an average improvement of 4%. There didn’t seem to be much difference resulting from different analyses used in generating the may-alias sets. There are several caveats to these results – they are heavily dependent both on the benchmarks used and on the optimization implementation. 


### Results: Limit Analysis

The authors then instrumented every load instruction in the benchmarks. A load was labeled as ‘redundant’ if the most recent load referenced the same memory address. They compared the number of redundant loads in both the unoptimized and optimized programs, and showed a significant improvement according to this metric (37% to 87% reduction in the number of redundant loads in the benchmarks). For the remaining redundant loads after optimization, the authors categorized them into several categories:

* **Encapsulation: **because of encapsulation, some redundant expressions were implicit in their IR, so they couldn’t eliminate them.
* **Conditional:** some loads are only redundant on certain control paths, not all.
* **Breakup:** the authors did not implement copy propagation, so redundant expressions consisting of smaller expressions weren’t eliminated. 
* **Alias failure:** issue with the analysis.
* **Rest:** undetermined.

The authors manually categorized the remaining redundant loads and determined that the vast majority were due to encapsulation. Conditional redundant loads were another significant source. The authors claimed to have no redundant loads resulting from an alias failure, but their ‘rest’ category was pretty significant and it is still possible for some redundant loads that fall into that category to result from an alias analysis failure.


### Extension: Incomplete Programs

The procedures and results mentioned by the authors all rely on the assumption that the entire program is available to the compiler. In many real-world settings, however, this may not be the case. For example, consider the following Modula-3 procedure definition: 


```
    PROCEDURE f(p: S1, q: S2) = …
```


In a type-unsafe language, if the compiler isn’t able to see every call of `f` in the program, then we have to assume that `p` and `q` may alias each other. If the language is type-safe, there are stronger assumptions we can make about the two variables leveraging the language’s type system.

The authors wrap up their paper with a brief discussion of how to modify their approach in the case of an incomplete program. First and foremost, for the AddressTaken component of their analyses, they add a second condition: if there exists a pass-by-reference formal, and some pointer shares the same type as it, then set the AddressTaken of that pointer to true. In this sense, the authors consider all available instructions for available code, and then conservatively consider the type system only for unavailable code. Additionally, the merging operation in their analyses is modified to also merge all available types that are related by the subtype relation. 

Interestingly enough, the authors claim that the open-world setting has an insignificant impact on the effectiveness of TBAA on their RLE.


## Discussion
* This paper presents a new kind of alias analysis: type-based alias analysis.
* The authors measured the new analysis algorithms in the context of its use: redundant load elimination
    * In addition to static analysis, the authors conducted dynamic runtime analysis and limit analysis, which measures against the measured best-case performance.
    * This paper is one of the first to utilize dynamic analysis and limit analysis, and highlights the importance of experimental results in respect to compiler optimizations.
* Shortcomings
    * We felt that the eight benchmark choices were limited. The authors write that the benchmarks were chosen because “other researchers have used several of them in previous studies.” Eight benchmarks cover nowhere enough use cases for programming languages that can express a plethora of algorithms. Ideally, we would have many more benchmarks across different use cases, and this paper did not have that. Granted, this paper came out in 1998 which could mean that there was a lack of Modula-3 benchmarks.
    * Why no copy propagation? Why not solve encapsulation?
* We also discussed where we can use limit analysis in our optimization measurements. We concluded that we can use limit analysis when we have an intractable optimization problem and its theoretically optimal runtime measurement is tractable.


## Related
* Redundant load elimination: [Introduction to load elimination in the GVN pass - The LLVM Project Blog](https://blog.llvm.org/2009/12/introduction-to-load-elimination-in-gvn.html)
* Steengaard’s Algorithm: [Points-to Analysis in Almost Linear Time](https://www.cs.cornell.edu/courses/cs711/2005fa/papers/steensgaard-popl96.pdf)
    * Available in LLVM in: [LLVM Alias Analysis Infrastructure — LLVM 18.0.0git documentation](https://llvm.org/docs/AliasAnalysis.html#the-steens-aa-pass)
* C++ Implementation for Statically Typed Lua
    * “Lua 5.3 was released on 12 Jan 2015”
    * Implemented in 2014
