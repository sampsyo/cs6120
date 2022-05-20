+++
title = "Peephole Improvements for Numyp with JAX"
[extra]
latex = true
[[extra.authors]]
name = "Aaron Tucker"
link = "https://www.cs.cornell.edu/~aarondtucker/"
+++

# Goal
Sometimes when writing code it is possible to run into numerical stability issues. These issues are typically dealt with through a mix of thoughtful fixes, painful debugging, or knowing various "tricks". What if your code did the tricks for you? This project's goal is to use "peephole improvements" to make it easier to write numerically stable code, rather than the end-user needing to know about them beforehand. Peephole improvements are based on peephole optimizations, but instead of trying to replace small parts of code with equivalent faster code they replace small parts of code with similar "better code" -- improving numerical stability changes the behavior of the program by avoiding NaNs in situations where the original wouldn't, but that is typically appreciated as long as it doesn't change other behavior too much.

# Implementation
The implementation of the project builds on several ideas explored in class, such as peephole optimizations and intermediate representations.

## Design
The basic design of the project is to convert code which uses numpy into an intermediate representation, perform the peephole optimizations on the intermediate representation, and then execute the code.

### Scope
The project's proof of concept for "peephole improvements" is to implement code which does the `logsumexp` trick for people. In machine learning the `logsumexp` operation will come up somewhat frequently, such as when computing the log of a softmax (when coupled with another normalization term). However, it is easy for this operation to run into overflow issues when any value of $x$ is big. The logsumexp trick of replacing $\log\sum_{i=1}^n\exp(x_i)$ with $\max_{i\in[n]}(x_i) + \log\sum_{i=1}^n\exp(x_i - \max_{i\in[n]}(x_i))$ with is a common way of dealing with this problem, since $\log\sum_{i=1}^n\exp(x_i) = c + \log\sum_{i=1}^n\exp(x_i-c)$ for all $c$, and so setting $c = \max_{i\in[n]}(x_i)$ guarantees that you only exponentiate terms $x_i - c \leq 0$, avoiding overflow issues. 

### JAX and Jaxpr
Thankfully, there is already a project which converts numpy code into an intermediate representation, then then executes it. [JAX](https://jax.readthedocs.io/en/latest/) is a project which combines autograd (automatic differentiation for python) and XLA to make it easier to write numpy code which runs on GPUs and lets you take derivatives automatically. This seemed like a better starting point for writing peephole improvements than trying to start from scratch because it already exists and has other useful features.

Jaxpr is JAX's intermediate representation for numpy code. While obviously different from BRIL, I found it to be pretty similar and pretty simple, so I was happy to build on top of Jaxpr rather than trying to target some other intermediate representation such as XLA or LLVM. Jaxpr, similar to BRIL, organizes programs into closures with a list of lines of code made of (basically) `(operation, in_variables, out_variables, type/shape)` tuples.

In order to use JAX, you need to import JAX's instrumented (among other things) version of `numpy` instead of using normal `numpy`. I've generally found this to be as easy as calling `import jax.numpy as np` and then forgetting about it.

## Hardest parts
There were a few snags that I hit, largely through my own ignorance, before I more systematically started working through JAX's very helpful documentation, especially [Autodidax](https://jax.readthedocs.io/en/latest/autodidax.html), [How JAX primitives work](https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html), and [Writing custom interpreters in Jax](https://jax.readthedocs.io/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html).

### JIT
The trickiest part of this project was when I was trying to understand the intermediate representation by tracing through and trying to directly understand the just-in-time compilation code. This was helpful for understanding some basic things about JAX, but made me spend a lot of time on the project without knowing about the existence of the much more directly helpful `make_jaxpr` and `eval_jaxpr` functions.

The most useful thing that I learned from trying to read the JIT code was that JAX's JIT compilation is based on tracing all of the uses of jax's version of numpy, and then rerunning code based on that trace. Incidentally, that brings us to the next hard part...

### Jaxpr only knows about Numpy
One issue is that Jaxpr only knows about the _numpy_ code that was run, not all the python code. As an interpreted language, Python is just running the code one line at a time. JAX doesn't reach into the interpreter to learn anything about the _python_ code, it only hears about things when you run its instrumented numpy code. This is enough to build automatic differentiation, target GPU computation, etc., but for example JAX does not learn about `if` statements.

To be more explicit, when you JIT compile code using JAX it waits until the first time the function gets used, then it traces everything that function does, then it builds a Jaxpr representation based on this code. But if for example the code had an if statement that changed which code gets executed, then  JIT compiled JAX code would still run whatever version of the trace it saw first. JAX can handle non-basic blocks, but only by using things like `jax.lax.cond`.

This was not a problem in implementing a logsumexp peephole improvement, but if for instance you wanted to check if each variable had NaNs (technically doable with `cond`), and then _print_ where the issue happened that would be impossible to implement as a peephole improvement without extending JAX's primitives.

# Evaluation
My evaluation for the project was pretty straightforward, since I was focusing more on claiming that the peephole improvement was correct, rather than showing that it's useful.

## Correctness
I wrote a bunch of [tests](https://github.com/atucker/jax-peephole/tree/main/tests) which check a few basic things.

1. The pattern matching for logsumexp fires when it can (even if there's code before, after, or during it), but not when the intermediate variables get used, or when the code is completely unrelated.
2. The logsumexp peephole improvement changes the code and produces the same answer in the most basic case, or if it's happening before or after other operations, and it doesn't break anything when the intermediate variables get used (even if it doesn't actually change the code in that instance).
3. The logsumexp peephole improvement [actually fixes](https://github.com/atucker/jax-peephole/blob/main/tests/peephole_fixes_logsumexp_overflow.out) an overflow issue! 🎉🎉🎉

## Usefulness

I think the main issue with this project is that I didn't implement many peephole optimizations. I also got started on trying to add the trick where you add a small identity matrix (i.e. `1e-6 * np.eye(d)`) to the input before doing matrix inversion, but the Jaxpr representation of `np.linalg.inv` was quite complex, and once I figured out how to find the inverses I still wasn't sure how to create my small identity matrix.

# Follow-up thoughts
While I was working on the project, I had a few other ideas.

## Where is this NaN coming from?!
Another thing that might be useful is just making it easier to understand where the NaN is even happening. Normally once I get a NaN in my code I spend a while trying to track down exactly where it started in order to understand what the issue is, but even tracking it down can take time. Wouldn't it be nice if you could add a decorator which when added just checks every intermediate step to see if that's where the NaN happened, and then tells you if it was?

While it might be tricky to do this as a peephole improvement because JAX doesn't have side-effectful code like printing, it might be a reasonable amount of effor to simply write a Jaxpr interpreter which performs this check automatically. Since you don't have to use this interpreter the rest of the time, this might be an okay implementation.

## Did I do the right thing?
While reading through more JAX documentation I thought more about the fact that it can output code to [XLA](https://www.tensorflow.org/xla), which seems to also output code to LLVM. Plausibly it would have been better for the project to work with XLA or LLVM instead, though working with Jaxpr was more targeted towards my original use case. I think this choice is somewhat defensible under the idea that once you're writing code in something like Tensorflow, you would simply call a `logsumexp` function (which can do all the stability tricks it wants), rather than calling `log` on `sum` on `exp` as you would when writing numpy code.
