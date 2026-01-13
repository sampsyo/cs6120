+++
title = "Musings on Aliasing in Rust for Optimization"
[extra]
bio = """
  Jeremy Ku-Benjet is a MS student at Cornell University.
"""
[[extra.authors]]
name = "Jeremy Ku-Benjet"
+++

## Experience Report
**Goal**: I planned to evaluate how well static analysis can do in finding aliasing pointers in the context of performance optimization by comparing the results of [Rupta's](https://github.com/rustanlys/rupta) static analysis to logs of Miri running on various Rust projects. Specifically I wanted to compare the number of aliasing pointers found by Rupta at compile time to the number of aliasing pointers found by Miri when run on various Rust project's tests. If these numbers were similar, it would suggest Rust is particularly well suited for alias analysis. If they are significantly different, it would suggest either alias analysis is hard in Rust, more similar to C for example, or more effort could be spent to leverage Rust's aliasing guarentees.

**What I Did**: I first implemented finding possible aliases using Rupta. Rupta performs a context sensitive analysis analysis to dump the memory locations, represented by MIR variables, that any pointer or reference can point to. To find the pointers which have a chance at aliasing each other, I reversed this relation, i.e. I dumped a map from memory locations to pointers. It is possible to do a better job at pointer analysis here. Simply reversing this relation does not take into account pointers which go out of scope before others are created. Verifying on small hand created tests this modification seemed to work. It's hard to verify the code on larger projects, but the numbers intutively seemed reasonable. Spot checking the functions which were noted to have lots of aliases also seemed reasonable.

I then modified Miri. Rust does not currently have a set of aliasing rules. Miri implements multiple models of aliasing rules, but the most popular is called [stacked borrows](https://github.com/rust-lang/unsafe-code-guidelines/blob/5854f2adf2081edaeabd77d1241365a5f6b4332a/wip/stacked-borrows.md). In short, for each memory location, Miri keeps track of a stack of items. Each item represents a pointer with different properties. There are then rules for when items are pushed and poped off the stack based on what references are created pointing to a given memory location and what operations are performed on these pointers. I recorded various metrics on stack sizes, in particular number of consecutive items of the same type and the total maximum stack sizes.

Running both of these implementation, I found the surprising result that Miri detected far more aliases than Rupta did. As an example, testing on the arena allocator [bumpalo](https://github.com/fitzgen/bumpalo) lead to Rupta finding 31 possible aliases and Miri finding 1000. I suspect this is an overesimate like Rupta's numbers as Miri keeps track of all references pointing to a given memory location even if they have no chance of aliasing each other. This problem may be exasterbated by Miri working dynamically, which means repeat calls to the same function aliasing some allocation will add different items to the stack.

**Success?**: Being upfront, this project failed. After seeing the discrepency in these metrics, I found their comparison was not meaningful. At best, they measured the same thing, serving as proxies for the amount of aliasing in a given Rust project. However, I don't know how to make a convicing argument for this metric being useful. For example, by these metrics it's possible for a project with many short lived, immutable alias pairs (the kind not breaking optimizations) to look similar to a project with long lived aliases between many pointers. A proper metric for measuring aliasing in the context of optimization would require some way to know if the aliases detected were important towards optimizations. That is, it would require using the aliases found by Rupta to fuel an optimization and comparing that to a similar optimization fueled by a profiling run of Miri.

To my knowledge, `rustc` doesn't perform any complicated alias analysis, meaning preforming an evaluation of this sort would require writing such an analysis. I didn't have time to do that.  However, even if I did have time, it likely would not be fruitful. This is because Rust's aliasing rules don't lead to complicated cases when it comes to optimizing code. In reality, it is effectively a binary where some pointers and references must be treated like C pointers with very few aliasing guarentees, and others are given extremely strong aliasing guarentees, making mutable aliases, the type preventing optimizations, undefined behavior. This can be cleanly lowered to LLVM by simply choosing when to add the `noalias` tag when lowering from MIR. LLVM then incorporate these assumptions in its alias analysis..

**What Now**: This report is embarrisingly light on fun implementations and emperical data, though the paragraph above hopefully explains the reason for the latter. The majority of time on this project went towards understanding alising in Rust, which is surprisingly complicated and ill defined. Given this, I'd like to conclude with some musings on aliasing. Much of these thoughts will already be common knowledge, though I'll try to bring an interesting throughline through them.

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

Examples of annotations to make mutable aliasing undefined also occur at the IR level. LLVM's [noalias](https://llvm.org/docs/LangRef.html#noalias) is a very similar keyword to `restrict`. `restrict` actually gets compiled down to `noalias` when using `clang`. When using `-O1` or higher, `clang` even optimizes `foo` to `foo_opt` (unrelated parts of the LLVM output removed for clarity):
```llvm
define i32 @foo(ptr noalias %0, ptr %1) #0 {
  store i32 4120, ptr %0, align 4
  store i32 6120, ptr %1, align 4
  ret i32 4120
}
```

## Aliasing in Rust
These optimizations, and in general simpler reasoning about the correctness of code[^3], are motivations for Rust's strict aliasing rules. It's worth restating the big ideas of these rules (the precise rules are complicated and currently undecided upon, though a popular model is [stacked borrows](https://github.com/rust-lang/unsafe-code-guidelines/blob/5854f2adf2081edaeabd77d1241365a5f6b4332a/wip/stacked-borrows.md)). In addition to C-like pointers, Rust has *references*. These are values which identify a memory location, like pointers, but unlike pointers the compiler enforces rules about their use. There are two types, immutable or shared references, `&`, and mutable references `&mut`. The compiler makes sure shared references are never mutated and makes sure mutable references pointing to a location are never read or written to after another reference pointing to that same location is read or written to. The difference can be seen in the example below:
```rust
// Shared references
let x = 1;
let shared_ref1 = &x;
let shared_ref2 = &x;
let _ = &shared_ref1; // Some valid reads
let _ = &shared_ref2;
let _ = &shared_ref1;
*shared_ref1 = 2;     // Rejected as shared reference `shared_ref1` cannot be mutated.

// Mutable references
let x = 1;
let mut_ref1 = &mut x;
let _mut_ref2 = &mut x;
let _ = *mut_ref1;      // Rejected as `x` was borrowed a second time making _mut_ref2.
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
This is the same LLVM output that the C code compiles to when using `restrict` (with an added `noalias` on `y` because it is also an `&mut`). It turns out this reasoning can be extrapolated, letting [most reference be marked as](https://github.com/rust-lang/rust/blob/cb79c42008b970269f6a06b257e5f04b93f24d03/compiler/rustc_ty_utils/src/abi.rs#L273) `noalias`. Though, a better way of saying this is the langauge's aliasing rules make sure not to preclude letting these optimizations occur.

## When Mutable Aliasing Occurs
Despite it being nice to not have to worry about mutable aliases, there are cases in which these guarentees cannot be easily applied. The most simple example is Rust's raw pointers. Raw pointers must exist, for example when doing FFI with C. To work raw pointers have to be C-like pointers, able to alias and be untracked by the compiler as `rustc` has no way to track what goes on in C code:
```rust
let x = 1;
let r1 = &mut x as *mut i32;
let r2 = &mut x as *mut i32;
let _ = unsafe { *r1 }; // Unsafe block required to dereference raw pointer `r1`
```
The compiler does not reject the above even though it is effectively the same thing as what happened using just mutable references as there is no guarentee on `*mut` not aliasing other `*mut`. One notable thing which pops up in this example is an [unsafe block](https://doc.rust-lang.org/nomicon/what-unsafe-does.html). This makes sense, as using raw pointers it becomes very easy to invoke undefined behavior. Consider the following code passing aliasing mutable references into the `foo` from above:
```rust
let x = 1;
let r1 = &mut x as *mut i32;
unsafe {
  foo(&mut *r1, &mut x);
}
```
`r1` and `&mut x` alias so making mutable references from them creates aliasing mutable references. This leads to undefined behavior, manifesting in an incorrect result when when calling `foo` as it gets optimized to `foo_opt` because it's args are `noalias`. Another way too look at this is `rustc` cannot attach `noalias` to the args of  `bar(x: *mut i32, y: *mut i32)` in the emitted LLVM.

Though, passing around raw pointers is probably unlikely to occur in most Rust code as dereferencing them is unsafe. It's more likely a safe abstraction allowing mutating shared references, *interior mutability* would be used. In Rust these are called `Cell`s. A function `bar` on two possibly aliasing mutable references is contrived, so a practical example of using this is creating a struct which you want shared references to but [whose operations cache themselves, requiring mutating some internal field](https://doc.rust-lang.org/std/cell/#implementation-details-of-logically-immutable-methods). The various `Cell`s are built off of the [primitive](https://doc.rust-lang.org/std/cell/struct.UnsafeCell.html) `UnsafeCell`. `UnsafeCell`s provide a method `get()` which gets mutable aliases to it's interior data. For example:
```rust
let x: &UnsafeCell<i32> = &4120.into();
let r1 = x.get();
let r2 = x.get();
// r1 and r2 are aliasing *mut i32s
```
The super power of `UnsafeCell` in the above is it is perfectly defined behavior to call `x.get()` and use the mutable aliases to shared memory it returns. However, with mutable alias to its memory, `&UnsafeCell` loses the ability to be treated like a normal shared reference and the compiler builds in special support for it. As with the above raw pointers, one way this manifests is when `&UnsafeCell` (or any of its derivatives like `Cell` or `RefCell`) is used as a function arg, it cannot be annotated with `noalias` when compiled to LLVM.

The interesting thing about `Cell`s and especially `RefCell`s is they end up forcing the compiler to treat them as possibly having mutable aliases despite them being safe types. Looking at [Rust's ABI code (same link as above)](https://github.com/rust-lang/rust/blob/cb79c42008b970269f6a06b257e5f04b93f24d03/compiler/rustc_ty_utils/src/abi.rs#L273), there are only two other cases in which `noalias` doesn't annotate function args:
```rust
let no_alias = match kind {
   PointerKind::SharedRef { frozen } => frozen, // frozen == true implies SharedRef is not an `UnsafeCell`
   PointerKind::MutableRef { unpin } => unpin && noalias_mut_ref, // noalias_mut_ref is a compiler flag manually stopping `noalias` annotations on MutableRefs
   PointerKind::Box { unpin, global } => unpin && global && noalias_for_box, // noalias_for_box is a compiler flag manually stopping `noalias` annotations on Boxs
};
```
The first case is simple. If `global` is `false`, `Box`s don't use Rust's default global allocator and instead using a [custom allocator](https://doc.rust-lang.org/beta/alloc/alloc/trait.Allocator.html). As allocators return `NonNull<T: PointeeSized>` which look to have similar aliasing guarentees to `*mut T`, that is having none, this implies `Box` can't have strong aliasing guarentees either. Finally, there is the case one of the mutable references (`Box` and `&mut`) doesn't implement isn't `unpin`. The reason for this looks to be a [poor interaction (bug?)](https://github.com/rust-lang/Miri/issues/3796#issuecomment-2299177277) between `Pin` and Rust's async/await.

Looking at these three case, there are limited ways mutable aliases can make their way into safe Rust code. Considering it's (in my experience so far) rare to change `Box`s allocator and the interaction with async/await seems [more like a bug being worked on](https://github.com/rust-lang/rust/issues/125735), these rules lets the programmer limit their view of mutable aliases to thinking about `Cell`s. They're effectively "choke points" where mutable aliases hide. They `Cell`s exist, but anecdotally, I find them to be a tiny minority of pointers. So, I find it cool to see the reasoning and potential performance penalty constained so much. 

[^1]: I'm taking this example from the [Stacked Borrows paper](https://dl.acm.org/doi/10.1145/3371109) which uses it to show a similar thing, though with different exposition.
[^2]: An interesting corrolary of this is that at compile time it's possible for pointers of the same type to have to be treated differently depending on how they were created. In other words, pointers have some extra data attached to them the compiler has to keep track of. This is sometimes called *provenance*. Ralf Jung has [two good](https://www.ralfj.de/blog/2018/07/24/pointers-and-bytes.html) [articles on this](https://www.ralfj.de/blog/2020/12/14/provenance.html) arguing for its existance. In some languages, for example Rust, provenance is in this interesting position where [it is not yet fully specified, but it still has to be reasoned about](https://doc.rust-lang.org/std/ptr/index.html#provenance).
[^3]: It's hard to give an nice self contained argument about this, but hopefully it should make sense with some thinking if you aren't already convinced. Another place to start is how this makes it easier to not do data races.
