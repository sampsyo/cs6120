+++
title = "What is a bug? New ways to test verification tools."
[extra]
latex = true
bio = "Max Fan is a first-year CS PhD student at Cornell University."
[[extra.authors]]
name = "Max Fan"
link = "https://max.fan"
+++

My project is driven by a deceptively simple question: *What is a bug in a
program verification tool?* 

In the following blog post, I will:
1. Explore a relaxed definition of a bug that ought to enable more effective
   testing of verification tools. 
2. Outline how to apply my definition to _abstractly test_ floating-point
   verification software.
3. Discuss my prototype implementation and preliminary results.

```
                                _______
                               /       \
                            ---|       |---
                               |  Bug  |
                            ---|       |---
                               \_______/
                                /  |  \
                               /   |   \
                              "    |    "
```


## What is a bug?
Consider a verification tool **VERIFY** to take some program *p(x)* and provide
a **GUARANTEE** such that for all inputs *x*, **GUARANTEE(**_p(x)_**)** holds.
At first blush, one might define a bug as follows: 

> <ins>Definition</ins> (_bug_). A bug in **VERIFY** exists iff there exists
an input _x_ such that **GUARANTEE(**_p(x)_**)** does not hold.

This is a perfectly reasonable definition. If you can violate a guarantee, it
seems like you have a bug.

But deep down in our hearts, there is _secretly_ a stronger notion of
correctness that we'd like the verification tool to respect. In particular, it
would be really bad if _I_ published a paper claiming my tool does **\$BLAH**
and prove **\$BLAH** to be a sound technique when in reality it does **\$BLEH**.
That sounds like a bug!

In other words: we'd like our verification tool to faithfully implement its own
theory (and respect its corresponding soundness proof). This is an entirely
reasonable demand for a user to have. I absolutely want my tool to do what it
claims to be doing.

Assuming the tool's theory is sound, we can define a new notion of bug, which I
call an _abstract bug_ (for reasons what will become apparent).

> <ins>Definition</ins> (_abstract bug_). An abstract bug in **VERIFY** exists
> iff it does not faithfully implement its own theory. 

Note that an _abstract bug_ is a stronger definition of bug. In particular, if
an abstract bug exists in a verification tool, there may not exist concrete
counterexamples $x$ such that **GUARANTEE(**_p(x)_**)** is violated.

## Interlude: Abstract Interpretation
To make this more concrete, consider a verification tool that relies on abstract
interpretation. To do abstract interpretation, we need the following three
ingredients:

1. A concrete domain 
2. An abstract domain corresponding to the analysis you wish to perform
3. A programming language syntax
4. A programming language semantics
5. Abstract transformers taking operations (from our programming language syntax
   and semantics), defined over our concrete domain, to be over our abstract
   domain 

satisfying certain soundness conditions.

A verifier (e.g. a sign analysis) will prove that for a program **p** and a
particular programming language semantics (e.g. IMP), there does not exist any
inputs **x** such executing **p(x)** under the semantics will violate the
results of the analysis.

The key idea we will exploit in this blog post is that there are many different
semantics (ingredient no. 4) for which the *same* concrete domain (1), abstract
domain (2), programming language syntax (3), and abstract transformer (5) are
still sound for!

## Abstract Testing of Floating-Point Software

With a fancy new definition, let's hunt for bugs! Observe that many (but not
all) verification tools operate over an abstract domain using overapproximate
simplifying assumptions. 

For example, the "standard model of floating-point error" assumes that each
floating-point operation $(op_{float} \ x \ y)$ can introduce up to $1+\epsilon$
error. To be precise, for every operation, the following overapproximation (of
the IEEE floating-point spec) holds for some unit round-off value $u$
(determined by the float precision and rounding mode used):

> $\exists |\epsilon| \leq u, (op_{float} \ x \ y) = (op_{real} \ x \ y) * (1 +
> \epsilon)$

Different tools may use other overapproximations, but the principle is the same:
to tractably verify you (typically) need to overapproximate in a way that admits
multiple implementations. That is, the above overapproximate model is sound to
verify against the IEEE floating-point semantics. It is _also_ sound to verify
against a defective floating-point semantics where we add up to $u$ error at
every single floating point operation.

So, how to test? Let's craft a deliberately defective implementation (that does
not comply with the IEEE floating-point spec) that our verifier is still good
for!

For floating-point verification, we can sample concrete inputs _and_
opportunistically construct bad $\epsilon$s at every floating-point operation to
violate the verification tool. Note that the IEEE floating point spec is fully
deterministic (which would constrain our test space if we were testing things
concretely), but our overapproximate, abstract error model is not (which lets us
adversarially fish around for bad $\epsilon$s).

In other words, our testing game is to find input $x$ and $\epsilon$-trace of
distinct $\epsilon_1$, $\epsilon_2$, $\epsilon_3$ ... added at to each
corresponding floating-point operation encountered at run-time such that we can
violate **GUARANTEE**. 

That's my project, in a nutshell! Every overapproximation provides a bigger
testing budget to smash the verification tool with. [^1] 

```

                                           Abstract Testing
                                              ======= 
                                              =     =
                                              =     =
                                              =======
                                                | |
                                                | |
                                                | |
                                                | |
                                                | |
                                                |_|

                                _______
                               /       \
                            ---|       |---
                               |  Bug  |
                            ---|       |---
                               \_______/
                                /  |  \
                               /   |   \
                              "    |    "
```


### Side quest: finding a good $\epsilon$-trace
A natural question to ask is: how do we find a good $\epsilon$-trace? To build
some intuition, consider a simple floating point program:

$$p := \frac{a - b}{c + d}$$

We wish to maximize the final error term in this program. Assuming that $a, b,
c, d$ are all non-negative, it suffices to:

1. Maximize the error term in the numerator ($a-b$). Recursing further, we
   realize we wish to
   - minimize the error term in $b$.
   - maximize the error term in $a$, and,
2. Minimize the error term in the denominator ($c + d$). Doing the obvious
   thing, we realize we need to
   - minimize the error term in $c$, and,
   - minimize the error term in $d$.

Once we know which direction to point each $\epsilon$, we can assign the most
extreme values (either $+u$ or $-u$) to get a pretty good $\epsilon$-trace.

For this program, the $\epsilon$-trace we computed happens to be the
**worst-case**, which is awesome. This technique generalizes in a pretty
straightforward manner to a backwards static analysis. [^2] 

## Implementation and Preliminary Evaluation
To implement this technique, I interpret a program twice:

- under an ideal (infinite-precision) semantics with no floating point error,
and,
- under an approximate semantics that adds a trace of $\epsilon$s at run-time.

To evaluate, I selected a few benchmarks from the
[FPBench](https://fpbench.org/benchmarks.html). I had already written a parser
and could reuse infrastructure for another project. My weapon of choice was
OCaml for similar practical considerations. [^3]


### Charts and Graphs
Below is a log-scale violin plot showing the distribution of absolute error
abstractly witnessed by my prototype tool, broken down by benchmark program. The
distribution comes from uniformly sampling from the program input space and
using the corresponding $\epsilon$-trace for each program constructed via the
backwards static analysis described above. [^4] (_Aside: I think more people
should use violin plots._) 

![Violin plot of "Absolute error (abstractly) witnessed by FPBench
benchmark".](/blog/sample-violins.svg)

Finally, here is a table showing the minimum and maximum error sampled on a few
selected benchmarks along with the corresponding guarantees provided by FPTaylor
and Daisy:

| Name | Min. error sampled | Max. error sampled | FPTaylor Guarantee | Daisy Guarantee |
|:----:|:-----------------:|:-----------------:|:-:|:-:|
| delta | 5.20879e-13 | 1.60481e-12 | 2.197089e-12 | 2.346965e-12 |
| delta4 | 3.11052e-14 | 6.98128e-14 | 7.676607e-14 | 1.160113e-13 |
| kepler0 | 2.28324e-14 | **6.66749e-14** | 3.463896e-14 | 1.044053e-13 |
| kepler1 | 6.96139e-14 | 2.21175e-13 | 3.689493e-13 | 4.806242e-13 |
| kepler2 | 5.27134e-13 | 1.57429e-12 | 2.199272e-12 | 2.464839e-12 |
| rigidBody1 | 1.11022e-16 | **3.38285e-13** | 2.948752e-13 | 2.948752e-13 |
| rigidBody2 | 1.11022e-16 | 3.21759e-11 | 3.574474e-11 | 3.606626e-11 |

As you can see, the maximum error sampled can get quite close to the guarantee
provided by the various tools. Entries that are **bolded** indicate a potential
bug, in either the verification tool under test (~thousands LOC) or the testing
framework itself (a few hundred LOC).

For future work, it would be cool to hunt down the bugs found and run this on
more tools and benchmarks. Additionally, a question not addressed by this tool
(and omitted in the blog post) is how to find a good input $x$. Currently, the
testing tool uniformly samples from the input space. One technique I wrote in my
initial proposal (but did not have the time to properly implement and evaluate)
is to use automatic differentiation. Future work could explore this further.

[^1]: For the sake of exposition and conceptual clarity, I am omitting various
    details on other overapproximations a tool might use and tricks like
[Sterbenz's lemma](https://en.wikipedia.org/wiki/Sterbenz_lemma) that complicate
the picture somewhat.
[^2]: A wrinkle occurs when terms can be shared. If a verification tool cannot
    reason about shared error terms, this algorithm still produces a worst-case
$\epsilon$-trace. Otherwise, this one might need to do a little searching to
find the worst-case $\epsilon$-trace. In particular, if error terms can be
shared, we start to care about the constants in the floating-point expression.
Concretely, `(x - 2x)` is quite different from `(2x - x)`; in the former we wish
to minimize the error term in `x`, while in the latter we wish to maximize the
error term in `x`. 
[^3]: The implementation and evaulation is at:
    https://github.com/Athena-Types/numerics-playground.
[^4]: To make the violin plot semi-legible and suitable for a blog post, some
    very small error samples are dropped. Specifically, the error values
that were equal to 0 once cast to a float (at the very end of testing) are
dropped from the plot. This only affects the plots for benchmarks `rigidBody1`
and `rigidBody2`. You may see the unadulterated range of error values sampled in
the table shown.
