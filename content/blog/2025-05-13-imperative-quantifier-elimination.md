+++
title = "Quantifier Elimination For Imperative Programs"
[extra]
bio = """
  Simon Bertron is pursuing a PhD in computer science at Cornell University. He mostly does theory-flavored systems research.
"""
latex=true
[[extra.authors]]
name = "Simon Bertron"
+++

# The Background

When people want to write mathematical statments formally, they make use of the quantifier symbols $\exists$ and $\forall$. Of course, many formally distinct mathematical statements are actually logically equivalent. For example, the following two logical formulas are equivalent (if $x$ is a real number):

$$
\exists x: a x^2 + b x + c = 0
$$

$$
b^2 - 4ac \geq 0
$$

It is worth noting briefly that the two formulas are only equivalent when $x$ is restricted to be a real number. If we allowed $x$ to be any complex number (or further restricted $x$ to be an integer!) the two formulas would no longer be equivalent. We say that the formulas are equivalent in the _theory_ of real numbers. For our purposes, a theory is a formal language together with some axioms defining which formulas constructed from that language are true.

When a formula can be written with and without quantifiers, the _quantifier-free_ version has some advantages from a computational perspective. The main advantage in the context of formal mathematical statements is that in most theories we care about, it is possible to describe an algorithm that decides the truth of any quantifier-free formula in that theory. This is not generally true of quantifier-ful formulas. If we squint and pretend our two formulas are computer programs, the quantifier-free version looks like a few arithmetic operations while the quantifier-ful version looks an awful lot like a loop that might never halt!

This suggests an algorithm for deciding the truth of a formula. First convert the formula to a quantifier-free form, then apply the universal-quantifier-free-deciding algorithm to the result. This is possible in some theories which have known procedures for removing quantifiers from any formula in that theory and, in fact, the Z3 SMT solver has a [quantifier elimination tactic](https://microsoft.github.io/z3guide/docs/strategies/summary/#tactic-qe) which attempts to apply such procedures.

Of course, quantifier elimination is not performing any decidability magic. Theories which admit quantifier elimination are fully decidable even by algorithms that do not perform quantifier elimination. So one reason Z3 has a quantifier elimination tactic (and not some other deciding algorithm) is because it can make the deciding _cheaper_. Even if our "loopy" pseudo-program above eventually terminates, it might take a long time. It would be easier to do the arithmetic instead.

# The Project

This brings us to the task at hand. If quantifier elimination can speed up SMT solving, maybe it can speed up regular old computer programs.

As we saw above, true quantifiers generally translate to loops that may never terminate. In most programming circles, writing loops that you know ahead of time might never terminate is generally frowned upon, so we will need more limited substitutes. We will use the commonly available `any` and `all` functions as our restricted quantifiers. `any` returns `true` if and only if _any_ of its inputs is `true` while `all` returns `true` if and only if _all_ of its inputs are `true` (get it?). This makes `any` an _existential quantifier_ ($\exists$) and `all` a _universal quantifier_ ($\forall$).

We will also restrict ourselves to a single theory: [Presburger Integer Arithmetic](https://en.wikipedia.org/wiki/Presburger_arithmetic). Formulas in Presburger Arithmetic are built from the following language: $[ \neg, \wedge, \vee, \exists, \forall ] \cup \mathbb{N} \cup [ +,-,<,= ]$. Where the first three symbols are the standard first order predicate logic _not_, _and_, and _or_.

We will operate on Python programs because Python has a few nice properties:

1. Python has an `ast` module in its standard library for parsing and manipulating Python programs so it should be easy to get started
2. Python has `any` and `all` built-in functions so it shouldn't be difficult to identify instances of "quantifiers" in Python code
3. There is _a lot_ of easy to find open source Python code in the world, so it should be straightforward to see if quantifier elimination can make any "real" programs faster

# [The Implementation](https://github.com/scober/cs6120-lessons/tree/main/project)

With $P_m$ removed, finding instances of Python code that can be quantifier-eliminated is straightforward. `ast.parse` gives us an abstract syntax tree and we search for sub-trees that contain only addition, subtraction, multiplication by constants (which we can safely treat as a shorthand for repeated addition), `==`, `!=`, `<`, `<=`, `>`, `>=`, `not`, `and`, `or`, `any`, `all`, and `range`. These allowed syntax elements are all carried over from the previous section, except for `range`, which takes the place of $\mathbb{N}$ and ensures we are only applying our quantifier functions to contiguous ranges of integers. An example of the sort of expression we are hoping to identify is

```Python
any(x < 16 for x in range(y, z))
```

and we are hoping to transform it into something like

```Python
y < 16 and z > y
```

##### A Brief Aside On A Technical Hiccup

When I started out, I was hoping to design my AST-rewriter/optimization pass as a function decorator that would re-write only decorated functions at runtime. Obviously (in hindsight), compiling a single function's hand-mangled AST outside of the context of its enclosing file is difficult to do at all and impossible to do correctly in general. So I pivoted to a design where I parse whole Python source files and write whole Python source files back out. Of course, performing interesting and correct static transformations on untyped Python source files is also impossible (depending on your definition of interesting), so I would call this redesign only a partial victory.

##### End Of The Brief Aside On A Technical Hiccup

Anyone who is curious about performing full quantifier elimination on Presburger Arithmetic formulas is welcome to take a look at [my source for the algorithm](https://amedvedev.ccny.cuny.edu/mathA4400s16/vandenDries.pdf). It is Theorem 4.4.2 and the pass I do in this project is a version of the one presented there.

The primary difficulty I had in translating a description of the algorithm in a math text to a piece of running code that modified computer programs was the irregularity of Python programs/ASTs. My final design ended up being a series of passes that applied more and more regularity to ASTs until I could perform a simple, final transformation. With that in mind, the rest of this section will be a high-level tour of the various regularizing passes the algorithm performs.

### Constant Folding

One huge source of irregularity in Python ASTs is that `1 + 1 + n` and `2 + n` are different ASTs. Worse, `0 + n` and `n` are different ASTs. I found it helpful to generate lots of unnecessary constants throughout the algorithm so I also found it helpful to do a lot of constant folding. One scenario where it is helpful to generate constants is when creating an AST for the sum of 0 or more existing AST subtrees. A nice way to do that is to create a `0` and then add a `+ <subtree>` for each subtree (and remove the `0` in a later constant folding pass).

The Python compiler performs constant folding if you ask it to perform optimizations, but I didn't know how to ask it to do only constant folding and nothing else. So I wrote my own simple folder and ran it early and often throughout my overall pass.

### Boolean Operators

A second source of irregularity in Python is the myriad equivalent ways that boolean values can be combined. So several subpasses were dedicated to enforcing the following invariants:

1. No `all`: `all(...)` is equivalent to `not any(not ...)`, so subsequent passes only have to reason about `any()`
2. No `not`: using DeMorgan's law and the fact that all comparison operators are negatable (i.e. `not x < y` is equivalent to `x >= y`), `not` can be effectively removed from our Python sub-syntax
3. No `or`: `any(p or q)` is equivalent to `any(p) or any(q)`, so any `or` at the top of the `any`'s internal AST can be "removed" by creating two smaller quantifier elimination tasks. Any `or` that is not at the top of the `any`'s internal AST can be moved to the top by putting that AST into conjunctive normal form.
4. No predicates that don't contain the _quantifier variable_. The quantifier variable is the variable that the `any` iterates over (i.e. `x` is the quantifier variable if the expression is of the form `any(... for x in range(...))`). Any predicate that does not contain the quantifier variable can be lifted out of the quantifier function in a loop invariant code motion like pass: `any(y < z and x < y for x in ...)` becomes `y < z and any(x < y for x in ...)`.

### Inequalities

Another source of irregularity in Python is that `i < n` and `n > i` are different ASTs. As are `x <= y` and `x < y or x == y`. So I dedicated several sub-passes to enforcing the following invariants:

1. The `range` bounds are explicitly enforced: `any(... for x in range(y, z))` becomes `any(... and x >= y and x < z for x in range(y,z))`. This is redundant at this stage, but will be critical when we remove the `range` call in the next step.
2. There are no `!=`, `x != y` is replaced with `x < y or y < x`
3. There are no `<=` or `>=`, `x <= y` is replaced with `x < y + 1` (remember that `x` and `y` are both integers)
4. There are no `==`, `any(x == <expr> and x < y for x in ...)` becomes `any(<expr> < y for x in ...)`
5. The quantifier variable appears only on the left side of inequalities and no other variable appears on the left side. Furthermore, all inequalities are scaled so that the same multiple of the quantifier variable appears on the left side (i.e. if `x` is the quantifier variable, `x < y and 2*x < z` becomes `2*x < 2*y and 2*x < z`).

### The Final Simple Pass

At this point we have a very regular AST. It looks like this:

```Python
any(m*x < <expr> and m*x > <expr> and ... for x in range(...))
```

The key insight is that of all the upper bound sub-predicates of the form `m*x < <expr>`, only one of them will matter. Only one of them will be the _least upper bound_. Similarly, only the _greatest lower bound_ will actually restrict what values of `x` return `True`. We don't know statically which upper bound is the least upper bound, because the upper bounds may have variables in them, but we can always check at runtime.

In a pure predicate logic setting, these special bounds can be "calculated" by statically unrolling a big loop that checks if each (lower bound, upper bound) pair is correct. But in Python we can just call `max` and `min`!

With the least upper bound and the greatest lower bound in hand, one only needs to check if there is a value between them that is divisible by `m` (remember that `m*x` needs to satisfy all of these inequalities, not `x`). The resulting program looks somthing like this:

```Python
min(...) - max(...) > m or 0 < min(...) % m < max(...) % m
```


# The Evaluation

### Correctness

Compiler passes (even those for class projects) should not change the behavior of the code they modify. This pass does not meet that standard for two reasons:

1. As I mentioned above, this is a static transformation of untyped Python code. There is no guarantee that all of the variables in the program are integers or even that `+` does what I expect. So the best we can hope for is that it is correct _for integer inputs_.
2. If I found all of the bugs in my implementation with testing I think that might be the first recorded instance of that happening in all of human history.

That being said, I did some testing! Because this pass operates on such a simple sub-language of Python, I built a random program generator and used those programs as my test inputs. I generated around 60 programs and when one of them failed (my pass changed the program behavior) I would whittle it down to a minimal failing program and keep that as its own test. Of the 60 randomly generated programs, around 5 failed and became minimized tests with special names. One even failed repeatedly and I was forced to minimize it to 2 _different_ minimal failing programs!

I found a lot of bugs this way, but the most frustrating one was a bug where my conjunctive normal form pass would create references to the same subtree from two different points in the AST. Then a follow-on pass would modify one of the subtrees and I would get spooky changes to portions of the AST that I was not expecting to be modifying.

### Performance

I chose to test the performance of my pass by testing big batches of function invocations since the running time of individual functions is very short and I was worried that overhead caused by i/o, garbage collection, or other interpreter behaviors would cause too much noise. Another benefit of batching is that it inherently averages running times over a large range of inputs (we will see soon that the running times of these programs can be _very_ sensitive to input size).

I tested the performance improvement of my pass by taking my randomly generated test programs and running them with significantly more inputs than I tested on (I found that for testing, an input range of $[-10,10)$ was sufficient, but for performance testing I did $[-10 000,10 000)$). I also made sure to generate my inputs in a different process than the one the tested function ran in and to print out the result of each quantifier function to prevent the Python compiler from optimizing away any of the function's work. I collected my numbers using [hyperfine](https://github.com/sharkdp/hyperfine), the file numbers start at 11 because the first 10 randomly generated programs were generated with slightly more restrictive parameters, all numbers are the result of 15 runs and are reported as _mean (standard deviation)_ in milliseconds:

| File # | Run Time (Original) | Run Time (Quantifiers Eliminated) |
|--------|---------------------|-----------------------------------|
|   11   |   297 (13)          |           271 (11)                |
|   12   |   317 (8)           |           307 (7)                 |
|   13   |   330 (6)           |           302 (3)                 |
|   14   |   47,308 (2,600)    |           282 (3)                 |
|   15   |   435 (8)           |           307 (7)                 |
|   16   |   217,818 (1,517)   |           278 (5)                 |
|   17   |   838 (16)          |           278 (3)                 |
|   18   |   311 (6)           |           304 (6)                 |
|   19   |   55,716 (200)      |           308 (3)                 |
|   20   |   37,889 (593)      |           370 (9)                 |

These numbers suggest that the quantifier elimination pass tends to have a slight positive improvement on performance and occasionally causes a massive improvement. With a little care we can identify fairly precisely when huge performance improvement spikes occur (and without having to parse the abomination that is randomly generated program number 14). Consider the following quantifier expression:

```Python
any(x > y for x in range(z))
```
we will repeat the experiment above with our sole input `z` ranging from `0` to `5000` and `y` set either to `0`, `2500`, or `5000`, all times reported in seconds:

|     y     | Run Time (Original) | Run Time (Quantifiers Eliminated) |
|-----------|---------------------|-----------------------------------|
|     0     |       0.2 (0.01)    |            0.2 (0.02)             |
|  500 000  |       2.1 (0.03)    |            0.2 (0.00)             |
| 1 000 000 |       2.6 (0.18)    |            0.2 (0.00)             |

Unsurprisingly, when a quantifier expression terminates early there is not much room for improvement. The real benefits of quantifier elimination come when it can eliminate long-running `any` and `all` calls.

In fact, randomly generated program 16 is pretty much the worst case scenario from this perspective:

```Python
any(n != n for i in range(n, -19*n))
```

The pass correctly transforms this to:

```Python
False
```

I think it is clear why those two expressions would have very different running times for, say, `n=-10,000` but have pretty similar running times for, say, `n=1`. I also hope it is clear that the huge time savings in the first table are mostly the result of the input programs being randomly generated and silly.

### Utility

One final measure of a compiler pass is how applicable it is to code that people actually write. To measure that, I gathered a bunch of open source Python code. I didn't gather it in any principled way, I just grabbed a bunch from the [GitHub trending page](https://github.com/trending/python?since=daily) and from [this collection of Python projects](https://github.com/vinta/awesome-python). 

According to [cloc](https://github.com/AlDanial/cloc), I searched ~1.68 million lines of Python code and found 0 lines that I could apply my quantifier elimination pass to. I think there are at least 3 obvious ways to expand the applicability of this pass (many thanks to [Professor Adrian Sampson](https://www.cs.cornell.edu/~asampson/) for thinking to do a text search and finding some programs that the pass does not apply to but probably could):

1. One obvious thing to do is to expand our arithmetic language. The [text](https://amedvedev.ccny.cuny.edu/mathA4400s16/vandenDries.pdf) that this algorithm comes from actually describes the algorithm for an expanded Presburger Arithmetic that includes a simplified `%` (modulus) operation, so that could be a good starting point. Though it is worth noting that quantifier elimination on full integer arithmetic is impossible, so this language expansion should be undertaken with care.
2. Once there is support for limited modulus, there is also a straightforward way to support three-argument `range()` calls. Snippets like
```Python
any(... for x in range(0, n, 2))
```
can be transformed into snippets like
```Python
any(... and x % 2 == 0 for x in range(0, n))
```
3. Finally, the pass can be more generous with what is considered an integer. The pass (as written) only counts variables and integer constants as integers. But lots of other things in Python are known to be integers! `len()` is an obvious example of something that could be safely treated as an integer in this context. As are many "numerical" functions like `min()`, `max()`, and `abs()`.

# Conclusion

In the end, this isn't a very useful compiler pass. But there are lots of places in the world where you are allowed to write useful compiler passes. I think the beautiful thing about CS 6120 is that it is a place where you can write use-_less_ compiler passes. I had fun and I learned something and I hope you did too!
