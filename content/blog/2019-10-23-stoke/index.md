+++
title = "Reimplementation of STOKE for BRIL"
extra.bio = """
Eashan Garg is an undergraduate studying CS and English.
"""
extra.author = "Eashan Garg"
extra.latex = true
+++

## Are optimizers good enough?

Traditional compiler optimizations target specific aspects of a program, such as dead-code or loops. While extremely effective, these techniques rarely find the *most* optimal sequence of instructions. What if we could somehow sample the space of all programs equivalent to the original, and select the one with the best performance guarantees?

## Introducing STOKE

While sampling the entire program space sounds great in theory, in practice it doesn't work so well. As programs get larger, the set of equivalent instruction-sequences grows exponentially, and evaluating them all becomes a daunting task. STOKE, a type of *superoptimizer*, attempts to solve this problem using MCMC, or random search techniques, to explore these programs. To guide its search path the algorithm uses a cost framework to measure how well a candidate program performs under our predefined heuristics.

There are a few limitations to this technique, most notably that it can only perform on loop-free sequences of code. For this specific implementation, I added a few additional constraints to make the sample space more manageable for my poor Windows laptop (enumerated upon in the following sections).

**Note**: I found this approach to be reminiscent of similarly thorny problem in machine learning - hyperparameter optimization. In the landmark paper "Random Search for Hyper---Parameter Optimization", the authors Bergstra and Bengio show that a random search is actually more efficient than a discrete, exhaustive search in high-dimensional spaces when the functions we are interested in are of low effective dimensionality. While it's hard to replicate the same optimization path each time, it's cool to see that these techniques seem to work surprisingly well.

*Constraining the sample space in order to make it easier to look through*
### Transforming programs

First, we need a way to actually generate samples from our program space. If we construct these programs completely at random, we risk 'bogo-sorting' for an unconstrained number of iterations. Rather, we can use a target program as our *base*, and apply small, guided transformations until the program is optimal. Lets consider three major transformations, each with equal probability of selection:

Original Bril Program:

```
main {
  a: int = const 4;
  b: int = const 2;
  c: int = const 3;
  d: int = add a b;
  print d;
}
```
#### 1. Drop a randomly selected instruction

Here, we insert a `nop` instruction to replace the line we want to delete here.

```
main {
  a: int = const 4;
  b: int = const 2;
  // Deleted instruction
  nop;
  d: int = add a b;
  print d;
}
```

#### 2. Swap two random instructions

As expected, we select two random instructions and swap their order within the program.

```
main {
  // Swap lines 1 & 2
  b: int = const 2;
  a: int = const 4;
  nop;
  d: int = add a b;
  print d;
}
```

#### 3. Replace existing instruction arguments

For this transformation, we replace the arguments, or operands of an existing instruction with new ones. Here, we apply a number of constraints to the replacement in order to ensure the program space is both reasonable and manageable:

1. Replacement arguments must be defined above the instruction in question
2. Values in `const` cannot be replaced. This is because the space of valid number values is enormous, and too difficult to sample.
3. Arguments are selected from a set of arguments with the same types.
4. `br` and `jmp` instructions are unavailable, as we are only working with loop-free sections of code.
5. As a simplifying assumption, `print` instructions cannot be replaced, and `ret` is unavailable (as there is only one function, and `print` has similar behavior in this use case).

```
main {
  b: int = const 2;
  a: int = const 4;
  nop;
  // A valid replacement for the add instruction
  d: int = add a a;
  print d;
}
```

Of course, there's a whole breadth of transformations that could also be applied to these instruction sequences: inserting instructions, replacing operators or even whole instructions, dropping chunks of code, etc. However, we'll stick with these three in order to avoid diluting our options at every step.

### Verifying correctness

The above transformations are applied at random, so there are no guarantees that the generated program will do what the original programmer intended, let alone be more performant. We start by verifying that this new representation replicates the exact same behavior as the original.

In the original STOKE algorithm, the authors separate this process into two distinct steps. First, a variable number of inputs is passed into both the original program and the transformed program, and the various side effects are compared. While this doesn't prove that these two programs are indeed equivalent, it's an efficient way to quickly weed out bad transformations. If the transformed program passes these tests, it is then passed into an SMT (Satisfiability Modulo Theories) solver. If we rewrite our two programs as series of SMT formulae, we can then pass them through a solver and have it verify their equivalence. The [Shrimp](https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/a-verification-backend/) project goes into detail with regard to the intricacies of implementing an SMT solver with Rosette (it can even cough up counter-examples!).

For correctness, we'll take a different approach. Repeated calls to the Bril interpreter is actually slower than making one call to Shrimp, so the benefit is minimal. Also, the Shrimp solver itself checks correctness by taking the intersection of live---variables from two basic blocks. While this works in most cases, suppose we take the following:

Original:
```
main {
  a: int = 5;
  b: int = 6;
}
```

Transformed:
```
main {
  a: int = 5;
  nop;
}
```

The intersection of live---variables, or `a`, is the same, so these programs will be considered equal. A possible notion of equality, but it isn't as applicable for this specific use case.

Instead, we take the simplest method for ensuring program equivalence. Since all Bril programs have only 'one' set of inputs, we can require every program to have effectual `print` instruction at the end. This is a reasonable assumption, as code without effects would be optimized away completely. Now, we can simply run both the original and transformed program through the Brili interpreter, and compare their outputs.

**Note**: Integration with Shrimp is still present in the implementation of STOKE for experimental value - a future improvement could be to make Shrimp more strict with program equivalence, which can then allow for STOKE to test programs with variable inputs.

### Evaluating performance

Now that we have a subset of equivalent Bril programs, we need to answer that second question --- which is the best? While comparing runtimes would be the most optimal heuristic, I argue that static instruction count is a more useful metric in this case, since it's easier to reason about improvements without having to worry about side-effects from the interpreter, optimizer, or underlying computer hardware. It is good to note that this rejects benefits that would arise from copy/constant propogation---like optimizations, as we only look at the instruction count itself, not the instructions that are being run.

Thus, each Bril program is assigned a cost equal to the number of instructions in the actual program. These are passed through a scoring function, $1 / e^x$, which maps these costs from 0.0 to 1.0. 

### Exploring the sample space

Pulling everything together, we implement a version of the Metropolis-Hasting algorithm. First, we take a sample program, apply a transformation, and score it as a sum of correctness (`0.1` if incorrect, `1.1` if correct) and performance. We then take the ratio of our new score and the original score. Sampling a random value from `0` to `1` from a uniform distribution, we can compare this to our ratio. If it greater than the ratio, we continue with the transformed program, otherwise we stay with our original program. This then continue for `n` iterations, and the program with the highest score is returned. Even though we occasionally make moves that impact the correctness of our program, we give STOKE the chance to find more interesting optimizations by fixing them on subsequent passes.

## How does it perform?

In order to evaluate the performance of Bril on real programs, we run it through a few larger benchmark programs --- and a series of smaller programs to do a more qualitative analysis of the superoptimizer's ability to find optimizations. Each 'optimal' program is generated with a depth of `n = 10`, and 100 of these programs are generated for each test case in sets of 20.

### Correctness

Based on the implementation of STOKE, we know that correctness is ensured as a side-effect of the verification step, so both our original program and optimized program are correct. Either way, we can run our two programs through the Bril interpreter once again, and ensure that they produce the same outputs.

### Optimization Performance

In order to test STOKE's ability to generate useful code, we evaluate on two separate types of benchmarks. First, we use simplified versions of the matrix multiply/polynomial multiply benchmarks provided by Wen-Ding Li as 'stress' tests for the algorithm. As expected, the large programs with 10+ instructions performed poorly --- STOKE is only able to find 2 - 3 instructions to optimize away, most of which was dead code to eliminate. These spaces are too large given a limited computational budget, although I expect they might perform slightly better given a much larger sample count. However, even with a large computational budget, I doubt there will be much to gain from increasing the depth of one given iteration, there are just too many possible transformations for the program to sample from.

On the other hand, STOKE performs surprisingly well given short <10 instruction sequences. Using the DCE, CSE, and CSE with commutativity examples from the CS6120 lecture notes, STOKE is able to consistently find an **optimal** code sequence with the least number of instructions within 10 - 15 iterations. Nice! It's especially exciting that the superoptimizer is able to perform these optimizations without explicit guidance, and with a small set of programs (~100 --- 150).

<img src="performance.png" style="width:100%"/>

(In the above figure, five batches of twenty runs each are averaged to produce STOKE's instruction count)

### Are Superoptimizers the future? It's hard to say

While our implementation of a stochastic superoptimizer is nowhere near as effective as `llvm -o3` or any other traditional series of compiler passes, it does perform much better than expected! Given a much larger computational budget and a better cost-model, perhaps we would be able to replicate the performance gains from the original STOKE paper. In particular, some concrete areas for extension:

#### 1. Use a better function to encode costs
The current implementation uses $1 / e^x$, whose values vanish as x grows past 10+ instructions. Rust is very restrictive when it comes to comparing floats, so having larger values here would be awesome.

#### 2. Add more transformations
The original STOKE paper explores a few more transformations than we implemented, such as operator replacement and instruction insertion. More available transformations would give the program many more 'moves' to make, and thus a larger space to sample from.

#### 3. Use a better cost-model
While static instruction count is a good baseline metric for programs, it still doesn't capture all the nuances of performance. For example, constant/copy propagation preserves the instruction count of a program, but replaces unnecessary `id` instructions with `const`. In addition, optimizations like constant folding can replace long sequences of operations with their `const` counterparts Perhaps assigning varying costs to individual instructions would help move towards these more interesting/complex optimizations.

### 4. Use a more complex distribution
For this implementation, we make the rather simplifying assumption that the proposal program distributions are symmetric, which allows us to define them as ratios of scores. However, this might not be the case - using a more complex formulation for the proposal distribution could result in better search paths.

For more details, check out Stanford's [STOKE](http://stoke.stanford.edu/) webpage, which houses many of the relevant papers for stochastic superoptimization.