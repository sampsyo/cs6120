+++
title = "Musings on Aliasing in Rust for Optimization"
[extra]
bio = """
  Jeremy Ku-Benjet is a MS student at Cornell University.
"""
[[extra.authors]]
name = "Jeremy Ku-Benjet"
+++

# Musings on Aliasing in Rust for Optimization
## Optimization using Aliasing
Compilers care about aliasing. Consider the following example C code[^1]:
```c
int foo(int *x, int *y) {
  *x = 4120;
  *y = 6120;
  return *x;
}
```
It would be great for the compiler to be able to optimize away the final memory read and simply return 4120, e.g.:
```c
int foo_opt(int *x, int *y) {
  *x = 4120;
  *y = 6120;
  return 4120;
}
```
However, that is unsound, if `x == y` then `foo(x, y) == 6120`, but `foo_opt(x, y) == 4120`. Generalizing this situation, the issue is `x` and `y` are *mutable aliases* for eachother, the pointers point to the same memory location and one is written to. Writes to one location might have the evil side effect of writing to the location of the other.

> Aside on Terminology: I and others in the world will frustratingly use the term "alias" to sometimes mean "mutable alias" and other times to simply mean *alias*, whenever two pointers point to the same location, no matter if that memory location is mutated through one of them. In this article, I will try to make it clear which definition I mean through context and being explict.

It would be nice to be able to check if `x != y`, but in C without some sort of intraprocedural analysis, this is totally impossible, the pointers don't carry information about where they came from across function barriers. All that can be done is for the programmer to tell the compiler `x` and `y` don't alias. In C, this can be done using the [restrict type qualifier](https://en.cppreference.com/w/c/language/restrict.html):
```c
int foo(int * restrict x, int *y) {
  *x = 4120;
  *y = 6120;
  return *x;
}
```
`restrict` makes it undefined behavior if there is an alias of `x` is modified in `foo`s scope[^2]. Therefore calls such as the below are undefined and the compiler can feel free to change the behavior of them when optimizing `foo`:
```c
int a = 1;
int b = foo(&a, &a);
```
And, it becomes valid to optimize `foo` to `foo_opt`. Yay! By letting this hard to optimize and hard to check at compile time cases be undefined behavior, language designers leave room for compiler designers to make code go faster.

Examples of annotations to make mutable aliasing undefined also occur at the IR level. LLVM's [noalias](https://llvm.org/docs/LangRef.html#noalias) is a very similar keyword to `restrict`. `restrict` actually gets compiled down to `noalias` when using `clang`. When using `-O1` or higher, it even optimizes `foo` to `foo_opt` (unrelated parts of the output removed for clarity):
```llvm
define i32 @foo(ptr noalias %0, ptr %1) #0 {
  store i32 4120, ptr %0, align 4
  store i32 6120, ptr %1, align 4
  ret i32 4120
}
```

## Aliasing in Rust
The optimizations and, in general, simpler reasoning about the correctness of code[^3] are motivations for Rust's strict aliasing rules. It's worth reviewing the big ideas of these rules (the precise rules are really complicated, see [The Rustnomicon](https://doc.rust-lang.org/nomicon/ownership.html) on ownership for more detail). In addition to C-like pointers, Rust has *references*. These are values which identify a memory location, like pointers, but unlike pointers the compiler enforces rules about their use. There are two types, immutable or shared references, `&`, and mutable references `&mut`. The compiler makes sure shared references are never mutated and makes sure mutable references pointing to a location are never read or written to after another reference pointing to that same location is read or written to. The difference can be seen in the example below:
```rust
// Shared references
let x = 1;
let shared_ref1 = &x;
let shared_ref2 = &x;
let _ = &shared_ref1;
let _ = &shared_ref2;
let _ = &shared_ref1;
*shared_ref1 = 2; // Rejected as shared reference `shared_ref1` Cannot be mutated.

// Mutable references
let x = 1;
let mut_ref1 = &mut x;
let mut_ref2 = &mut x;
*mut_ref2 = 2;
let _ *mut_ref1; // Rejected as `mut_ref2` was modified previously.
```

The extra information attached to references gives the compiler significant extra information to prove things about the program. For example consider rewriting `foo`: 
```rust
fn foo(x: &mut i32, y: &mut i32) -> i32 {
  *x = 4120;
  *y = 6120;
  *x
}
```
the compiler knows `x` cannot have any mutable aliases in before it is read at the bottom of the function (notably it can't alias `y` in the line `*y = 6120`) and so it can optimize the code:
```llvm
define i32 @foo(ptr noalias %0, ptr noalias %1) #0 {
  store i32 4120, ptr %0, align 4
  store i32 6120, ptr %1, align 4
  ret i32 4120
}
```

TODO: POINT OUT THE CODE AND SHOW NOALIAS IS ALWAYS A THING EXCEPT FOR THE WEIRD INTERIOR MUTABILITY PINNING THING, THOUGH MAYBE IGNORE THAT FOR NOW. THEN SHOW SOME NUMBERS WHICH SHOW THAT ALIASING STILL IS A THING BY RUNNING BUMPALO AND AHO-COSICK AND MAYBE A COUPLE OTHER THINGS. CONCLUDE WITH SAYING ALIASING IS A THING AND HAS TO BE BUT WITH STRICT ALIASING RULES WE CAN STILL GET STRONG GUARENTEES.

TODO: WRITE UP THE GENERATOR THING AS AN EXAMPLE OF WHY WE CAN'T HAVE NOALIAS EVERYWHERE. SHOW   ADD SOME NUMBERS TO SHO... ACUTALLY THIS IS PROBABLY A TANGENT.

It 

This is great, but also totally infeasible. Performant code often requires mutable aliasing to the same pieces of memory. Consider a arena allocator

TODO: The argument I want to make is there is significant aliasing going on in rust programs. However, despite that, the strict rules about references make that okay and more generally there is a lot of perfectly fine aliasing and its actually a pretty small subset of aliasing which breaks stuff. In particular, writing to aliased data without some sort of strong ordering (basically a analogous to concurrent programming stuff almost). Evaluation is used to back up the first statement, both with Miri and rupta which measure slightly different things. Rupta measures the obvious "what points where," alias relation and Miri is less direct but gives further information on kind of what types of aliases are happening (lots of shared references or lots of unsafe cells or something).

[^1]: I'm taking this example from the [Stacked Borrows paper](https://dl.acm.org/doi/10.1145/3371109) which uses it to show a similar thing, though with different exposition.
[^2]: An interesting corrolary of this is that at compile time it's possible for pointers of the same type to have to be treated differently depending on how they were created. In other words, pointers have some extra data attached to them the compiler has to keep track of. This is sometimes called *provenance*. Ralf Jung has [two good](https://www.ralfj.de/blog/2018/07/24/pointers-and-bytes.html) [articles on this](https://www.ralfj.de/blog/2020/12/14/provenance.html) arguing for it's existance. In some languages, for example Rust, provenance is in this interesting position where [it is not yet fully specified, but it still has to be reasoned about](https://doc.rust-lang.org/std/ptr/index.html#provenance).
[^3]: It's hard to give an nice self contained argument about this, but hopefully it should make sense with some thinking if you aren't already convinced. Another place to start is how this makes it easier to not do data races.
