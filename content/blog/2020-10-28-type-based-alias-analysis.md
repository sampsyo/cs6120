+++
title = "Type-Based Alias Analysis"
[extra]
bio = """
  Kenneth Li is a first-semester M.Eng. student who is interested in functional
  programming. He sings bass in the 
  [Cornell Hangovers](http://www.hangovers.com/).
"""
[[extra.authors]]
name = "Kenneth Li"
+++

Alias analysis allows compilers to determine which pointers may (or must) 
refer to the same memory location. This is most useful for the purpose of 
instruction reordering; if a compiler knows that two memory instructions refer 
to different memory locations, it can switch the order of the instructions. In 
_Type-Based Alias Analysis_, Diwan, McKinley, and Moss propose and evaluate 
alias analyses for Modula-3 (a type-safe language with in heritance) that use 
type information instead of assignment tracking. Each analysis successively 
refines the previous: _TypeDecl_ uses only type/subtype compatibility, 
_FieldTypeDecl_ additionally considers field names and other properties, and 
_SMFieldTypeRefs_ augments the former with a single pass through the 
instructions to pick up assignment information. The analyses all do a may-alias 
analysis; all alias pairs calculated are possible aliases, but all other pairs 
are confirmed to not be aliases.

## Algorithms
The _TypeDecl_ analysis relies on a simple predicate: two referenced memory 
access paths `p` and `q` may be aliases if and only if there exists a type that 
subtypes both the type of `p` and the type of `q`. If such a type did not 
exist, for example, then if `p` and `q`were aliases, the object they refer to 
would have two different, incompatible types. _FieldTypeDecl_ takes this 
further by adding checks for field names, array accesses, and pointer 
dereferences. For example, paths `p.f` and `q.g` may be aliases only if the 
field names `f` and `g` are exactly the same and `p` and `q` themselves may be 
aliases. Additionally, a pointer dereference `q^` may alias a field `p.f` or an 
array access `p[i]` only if its address was ever taken by the program (since 
otherwise no pointer to it could exist). Further checks exist for comparisons 
between pointers and array accesses, which the authors sum up as follows:

1.	Identical [access paths] always alias each other.
2.	Two qualified expressions may be aliases if they access the same field in 
potentially the same object.
3.	A pointer dereference may reference the same location as a qualified or 
subscripted expression only if their types are compatible and the program may 
take the address of the qualified or subscripted expression.
4.	See (3)
5.	In Modula-3, a subscripted expression cannot alias a qualified expression.
6.	Two subscripted expressions are aliases if they may subscript the same 
array. _FieldTypeDecl_ ignores the actual subscripts.
7.	For all other cases of [access paths], including two pointer dereferences,
_FieldTypeDecl_ uses _TypeDecl_ to determine aliases.
Finally, _SMFieldTypeRefs_ uses a simple flow-insensitive pass through the 
program’s instructions to merge types that might become aliases. It uses a 
table that tracks the equivalence classes for a given type (such that for a 
type `T` with entry `{T1, T2, …}`, an access path of type `T` may be a 
reference to any `T1, T2, …`). Upon encountering an assignment from type `T1` 
to a pointer of type `T2`, it adds `T1`’s entry to `T2`’s equivalence class. 
Notably, this is an asymmetric operation; the assignment does not change `T1`’s 
equivalence class. This is a more refined version of _TypeDecl_ above, so by 
replacing _TypeDecl_ with lookups from this table (called _SMTypeRefs_ in the 
paper, for “Selectively Merge Type References”), we have the final algorithm 
_SMFieldTypeRefs_. 

## Evaluation
The paper does a surprisingly thorough evaluation job considering the paper 
was written in 1998. The authors observe that previous alias analyses were 
evaluated by their static properties, like the size of the set of alias pairs, 
and their dynamic properties, such as the impact of an alias analysis on a 
compiler optimization that uses it. The static evaluation, performed over a 
selection of 10 benchmarks, finds that _TypeDecl_ performs much worse than 
_FieldTypeDecl_ in terms of alias set size, but _SMFieldTypeRefs_ barely 
improves on _FieldTypeDecl_ at all. It also shows that applying type-based 
alias analysis interprocedurally generates huge numbers of aliases, rendering 
this type of analysis infeasible for global optimization. 

The authors then proceed to measure the impact of the three different analyses 
on redundant load elimination, which moves invariant memory references out of 
loops. They found that the number of redundant loads removed did increase with 
the power of the analysis and that the amount of optimization improvement 
between analyses was correlated with how much smaller the alias set became in 
the static evaluation. On the other hand, when they considered the execution 
time post-redundant-load-elimination, they found that all three analyses 
performed similarly, averaging a 4% speedup from RLE. Thus, perhaps 
counterintuitively, more precision in the alias analysis doesn’t necessarily 
yield significant gains in runtime speed.

However, the authors note that static properties don’t directly correlate with 
performance benefits, and don’t lend themselves to comparing two different 
alias analyses – in both cases, it’s hard to tell whether the disambiguated 
pointers will be relevant for the intended use case. Dynamic properties, on the 
other hand, suffer from overspecificity – it’s difficult to evaluate general 
efficacy of an alias analysis from performance improvements on a few specific 
optimizations and benchmarks. Worst of all, neither evaluation can give an idea 
of how much better an analysis could be.

To alleviate these problems, the paper introduces the concept of limit 
evaluation to figure out whether there were missed optimization opportunities 
due to undetected aliases. The authors instrumented their benchmarks to record 
the address and value of every load, allowing them to realize exactly how many 
aliases there actually are in an execution. This extra step indicated that the 
TBAA-assisted RLE removed between 37% and 87% of redundant loads, and for 6 out 
of 8 of the benchmarks only 5% of the remaining loads were redundancies that 
could be eliminated. Pushing further, the authors even noted that, in the other 
two benchmarks, the majority of the remaining loads were due to limitations in 
the RLE implementation, and not a single missed opportunity was due to TBAA 
failing to disambiguate references. 2.5% of the remaining loads were due to 
unknown causes, making that an upper bound for the possible improvement.

Ultimately, the evaluations show that though it is easy to come up with 
examples in which TBAA fails to properly differentiate references, in practice 
it has very little room for improvement with respect to redundant load 
elimination. By evaluating on four metrics, the authors were able to draw more 
nuanced conclusions; for example, a runtime-only evaluation might conclude that 
_TypeDecl_ is sufficient, but _FieldTypeDecl_ actually yields significantly 
more opportunities for RLE. The authors also do not state their results in a 
vacuum; they clearly state that the results are only with respect to one 
optimization (RLE) and the set of benchmarks they used. However, benchmark 
diversity and, in this case, optimization diversity is still lacking; the 
evaluation metrics reveal deep insight about this case, but are not broadly 
applied.

The significance of this work is not to be understated; alias analyses over 
types instead of instructions run much more efficiently due to the vastly 
reduced search space, and the results in paper indicate that the precision 
tradeoff could be minimal for some applications. From a modern perspective, 
it seems that compiler developers have come to the same conclusion; many 
modern compilers implement and support type-based alias analyses for many of 
their optimization passes. Among these compilers are GCC and LLVM, showing that 
the benefits have even extended outside the paper’s original realm of type-safe 
languages. 
