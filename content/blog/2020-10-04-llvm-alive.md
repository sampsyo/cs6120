+++
title="Provably Correct Peephole Optimizations with Alive"
+++

## Contributions

This paper introduces Alive, a Domain-Specific Language designed for conveniently verifying LLVM optimizations. The Alive language provides a framework for encoding LLVM transformations of compiled code. The implementation uses a transformation encoding to generate preconditions and correctness conditions expressed in terms of predicates, which may then be automatically verified by an SMT solver. The result is a verified LLVM optimization in Alive, which also provides a tool for extracting the optimization into C++ code. As an additional feature, the language supports built-in predicates which are implemented by dataflow analysis.

The main difficulty with verifying LLVM opmtimizations is that the source code may have undefined behaviors. For this reason, much of the interesting design decision for Alive are related to handling undefined behavior. In principle, Alive is intended to check that optimizations are *refining*, that is, that the set of behaviors of the target code is included in the set of behaviors of the source.

After introducing Aive, the paper turns toward more technical concerns. In particular, it proves two versions of a soundness theorem for Alive's verification method--it first considers the language without memory and pointer concerns, then re-introduces pointer arithmetic. The theorem in essence states that if the Alive back-end accepts a given optimization, then the code transformation is semantics-refining.

## Detailed Summary

At a high-level, the process of verifying a peephole optimization in Alive is as follows:

1. Specify the transformation in Alive's domain-specific language (DSL).
2. Alive discharges SMT formulas that encode the semantics of LLVM operations and datatypes.
3. The formulas from step 2 are used as verification conditions and passed to an SMT solver.
3. SMT solver either verifies the optimization as valid or returns a counterexample showing why the optimization is wrong.
4. (Optional step) If optimization is correct, Alive synthesizes C++ code that can then generate the relevant optimization to optimize LLVM code.

### Alive's DSL

Optimizations in Alive are written as a transformation from a source (left hand side) to a target (right hand side). An example optimization is given below.

```
Pre: C1 & C2 == 0 && MaskedValueIsZero(%V, ∼C1)
%t0 = or   %B,  %V
%t1 = and %t0,  C1
%t2 = and  %B,  C2
%R  = or  %t1, %t2
=>
%R  = and %t0, (C1 | C2)
```

Here the left hand side is everything before the `=>` and the right hand side is everything after. The DSL abstracts LLVN semantics: anything beginning with a capital C is a constant, the pattern %t[num] are temporary registers and the lack of data-types indicates that this transformation is valid for all LLVM datatypes. The keyword `Pre:` allows the user of the DSL to specify a pre-condition which abstracts the results that an LLVM compiler may infer from dataflow analyses before applying a transformation. These predicates are hard-coded into Alive.

### Correctness of Optimizations

The goal of Alive is to prove that a target optimization refines the behavior of a source program under the presence of undefined behavior.  Undefined behavior is the result of the assumptions a compiler makes about certain instructions of a program. When an instruction with undefined behavior is used then the compiler may replace it with an arbitrary sequence of instructions or assume that such undefined behavior never occurs. For example, in `C` the instruction `x + 1 > x` can be replaced with `true` (i.e. any value `!= 0`) as signed overflow is undefined behavior.

*A compiler should never introduce new behavior when there is no undefined behavior but can produce new results in the presence of undefined behavior.*

A nice summary of Alive and a discussion on undefined behavior can be foundin this [blog post](alive-blog) by the authors of the paper.

#### undef and poison

In LLVM there are three types of undefined behavior. The first arises from the keyword `undef`. `undef` in LLVM represents any value (given a specific width) each time it is read from. For example, in the program

```
%z = xor i8 undef, undef
```

`%z` can be any value in the range `{0, ..., 255}` as we are taking the xor of any two 8-bit values. A more interesting example is

```
%z = or i8 1, undef
```

where the value of `%z` becomes any odd integer in the range `{0, ..., 255}`. `undef` allows the compiler to aggresively optimize a program as LLVM [makes the assumption](undef-val) that whenever an `undef` is seen "the program is well-defined no matter what value is used."

`poison` values are distinct from `udef` values as they are used to indicate that "a side-effect-free instruction has a condition that produces undefined behavior." There is no way to explicitly indicate that a value is `poison` in LLVM. As an example, the LLVM instruction 

```
%r = shl nsw %x, log2(C1)
```

causes `%r` to be a `poison` value as the left shift might cause `%x` to go from a positive value to a negative one. If the value of `%r` were to be used in an instruction with side-effects, such as memory stores, then we get true undefined behavior. Furthermore, `poison` values will taint any subsequent dependent instructions meaning `poison` is propagated throughout a program. `poison` values are deferred undefined behaviors and can only be identified through careful analyses.

Finally, instruction attributes allow the compiler to make assumptions about certain instructions. For example, `nsw` means "no signed wrap" which makes signed overflow undefined allowing us to perform the optimization where `x + 1 > x` replaced with `true`.

### Refinement

By considering the types of undefined behavior in LLVM, the authors define what valid refinements of programs are. If there is an `undef` value then an optimization must produce a value that is subset of the `undef` value produced by the source as `undef` represents a set of possible values. Similarly, if the source program has a `poison` value then the target program may have a `poison` value but a target instruction cannot create `poison` values when there were none in the source program. So, the authors use the intuition that compilers should not introduce new undefined behavior to generate SMT formulas that capture the following correctness criteria:

> (1) the target is defined when the source is defined,
> (2) the target is poison-free when the source is poison-free, and
> (3) the source and the target produce the same result when the source is defined and poison-free.

The SMT formulas generated by Alive encode all possible types for the instructions in the source and target programs and also encode `undef`, `poison` and instruction attributes to model undefined behavior.

### Example

The optimization of replacing `(X << C1) / C2 ` with `X / (C2 >> C1)` whenever `C2 is a multiple of C1` can be written in Alive as follows

```
Pre: C2 % (1<<C1) == 0
%s = shl nsw i4 %X, C1
%r = sdiv %s, C2
  =>
%r = sdiv %X, (C2 / (1 << C1))
```

This optimization does not refine the source program when `C1 = width(C1) - 1` as `X << C1` may overflow. This is not obvious to see so Alive produces the following output 

```
ERROR: Mismatch in values of i4 %r

Example:
%X i4 = 15 (0xf)
C1 i4 = 3 (0x3)
C2 i4 = 8 (0x8)
%s i4 = 8 (0x8)
Source value: 1 (0x1)
Target value: 15 (0xf)
```

which gives a counterexample for `4`-bit unsigned integers. Here, the issue is that the optimization causes the target to produce a different result for a specific input that causes no undefined behavior in the source program. Note that the source program uses `nsw` to indicate that signed overflow is undefined behavior but assumes that unsigned overflow is not undefined behavior. This is why the counterexample produced by Alive uses unsigned integers.

The authors of alive opened a [bug report](pr21245) and a fix was accepted where the pre-condition of the optimization was strengthened to 

```
Pre: C2 % (1<<C1) == 0 && C1 != width(C1)-1
```

## Alive's Impact

At the time of publication, Alive's authors manually translated 334 LLVM peephole optimizations (InstCombine) to Alive out of a possible 1028 instructions, meaning 694 were not processed for verification. Out of the translated optimizations, the authors found 8 bugs where the most uncommon bug was due to the introduction of undefined behavior. The authors state that most of the time Alive runs in a few seconds while for instructions with multiplication and division it "can take several hours or longer to verify the larger bit-widths" as most SMT solvers struggle with such inputs. The remaining optimizations could not be translated as they include instructions that were not supported by Alive at the time.

In addition to verifying existing optimizations, the authors also created a LLVM+Alive version of the LLVM compiler where the optimization verified by Alive were replaced by the C++ code generated by Alive. On average, code generated by LLVM+Alive was 3% slower than LLVM's -O3 (the most aggressive optimization option) despite covering a fraction of the optimization LLVM offers.

Interestingly, the authors also recorded the number of LLVM optimizations that were used when testing LLVM+Alive on the SPEC benchmark. Optimizations covered by Alive were used 87000 times and only a small number of optimizations were used. 159 of the 334 were used in some way and the top ten of those optimizations account for almost 70% of the total invocations. The figure given in the paper is shown below.

![fig9](alive-fig9.png)

The authors also state the following in the introduction.

> [...] we have prevented dozens of bugs from getting into LLVM by monitoring the various InstCombine patches as they were committed to the LLVM subversion repository. Several LLVM developers are currently using the Alive prototype to check their InstCombine transformations.
>

### Alive2 (Detour)

Following this work, Alive's authors have improved on Alive in multiple ways including [floating-point support](alive-fp), a precondition [inference](alive-infer) tool for optimization and the formalization of Alive, called [AliveInLean](alive-lean), in the [Lean](lean) theorem prover (this work assumes that "proof obligations are correctly discharged by an SMT solver"). Most recently, the authors of Alive have switched Alive to [maintenance mode](alive-git) and introduced a newer version of Alive, called [Alive2](alive2-git). A nice introduction to Alive2 can be found in a series of blog posts ([1](alive2-blog1), [2](alive2-blog2), [3](alive2-blog3)). Alive2 supports regular LLVM code in addition to the DSL of Alive along with bidirectional verification (the => of Alive can be replaced with <=>). So far, Alive2 has found 58 total [bugs](alive2-bugs) in LLVM & [Z3](z3).

### Contributions to LLVM (Detour)

Some of the authors of this paper (everyone except Santosh Nagarakatte) have pushed for the [removal of undef](remove-undef-llvm) from LLVM and introduce a new construct they call freeze. freeze was [added](freeze-twitter) to [LLVM](freeze-llvm) but undef was not removed.

The most interesting aspect for this work is Regher's blog post on why (undefined behavior is not always unsafe programming)[undef!=unsafe]. In LLVM, the [undef](undef-val) is used to indicate that "the program is well-defined no matter what value is used" which gives an optimizer the freedom to optimize the program. Regher argues that undefined behavior at the programmer visible abstractions level allows for more efficient programs and simpler compilers at the cost of program correctness. On the other hand, undef in LLVM "is an internal design choice" that need not be visible for the programmer to allow for better optimizations. If error-checking can be done at the higher-level and we can conclude that a program does not need such checks, then undef can be inserted into code at the LLVM level which then means these error checks can be factored out and more optimizations can be applied. 

In general, computer science education considers undefined behavior a harmful concept. However, we also contend with the fact that undefined behavior might be required for aggressive optimizations and speculative execution. Alive seems to take a third route as articulated in Regher's blog post and documented in the LLVM documentation: undefined behavior is ok as long as it "refines' ' the original (source) program in some way.

> Undefined behavior is the result of a design decision: the refusal to systematically trap program errors at one particular level of a system. The responsibility for avoiding these errors is delegated to a higher level of abstraction.
> ...
> The essence of undefined behavior is the freedom to avoid a forced coupling between error checks and unsafe operations.


[alive-blog]: https://blog.regehr.org/archives/1170
[alive-fp]: https://link.springer.com/chapter/10.1007/978-3-662-53413-7_16
[alive-infer]: https://dl.acm.org/doi/abs/10.1145/3062341.3062372
[alive-practical]: https://dl-acm-org.proxy.library.cornell.edu/doi/abs/10.1145/3166064
[alive-lean]: https://link.springer.com/chapter/10.1007/978-3-030-25543-5_25
[lean]: https://leanprover.github.io
[alive-git]: https://github.com/nunoplopes/alive
[alive2-git]: https://github.com/AliveToolkit/alive2
[alive2-blog1]: https://blog.regehr.org/archives/1722
[alive2-blog2]: https://blog.regehr.org/archives/1737
[alive2-blog3]: https://blog.regehr.org/archives/1837
[alive2-bugs]: https://github.com/AliveToolkit/alive2/blob/master/BugList.md
[z3]: https://github.com/Z3Prover/z3
[remove-undef-llvm]: https://lists.llvm.org/pipermail/llvm-dev/2016-October/106182.html
[freeze-twitter]: https://twitter.com/johnregehr/status/1191765816422760448?lang=en
[freeze-llvm]: https://github.com/llvm/llvm-project/commit/58acbce3def63a207b8f5a69318a9966
[taming-undef-behav]: https://dl.acm.org/doi/abs/10.1145/3140587.3062343
[undef!=unsafe]: https://blog.regehr.org/archives/1467
[undef-val]: http://llvm.org/docs/LangRef.html#undefined-values
[pr21245]: https://bugs.llvm.org/show_bug.cgi?id=21245