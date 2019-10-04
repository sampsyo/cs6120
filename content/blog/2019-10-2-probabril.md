+++
title = "Probabril"
[extra]
latex = true
bio = """
bio
"""
[[extra.authors]]
name = "Oliver Richardson"
link = "https://www.cs.cornell.edu/~oli/"  # Links are optional.
[[extra.authors]]
name = "Michael Roberts"
+++


# Introduction

Often one would like to represent a non-deterministic process as a combination of operations. Such a representation is often much more compact and often much more clearly represents the process you have in mind, and is easier to edit; a probabilistic program is one such way of representing such a program. All that was needed to do to turn `bril` into a probabilistic programming language is to add a source of randomness --- in our simple case, a coin flip operation. Running such a program, gives you a sample from the distribution.

Of course, having the source code, we can do much more than running programs.

## The Goal: An Exact Solver

The main bit of the project was to write an abstract interpreter that solves for the exact distribution that a probabilistic program represents. For instance, the program below flips two coins:

```
  x : bool = flip;
  y : bool = flip;
  ret
```
we should be able to see represents the distribution
\[
p \left(\begin{matrix}x \land y  \\
x \land \lnot y  \\
\lnot x \land y \\\lnot x \land \lnot y  \end{matrix}\right)
 = \left(\begin{matrix} .25 \\ .25 \\ .25 \\ .25 \end{matrix}\right)
\]

By repeatedly running a program and recording the frequencies of resulting environments (Monte Carlo sampling), one can get a rough approximation of the distribution, but this can be less than satisfying for a number of reasons:

- The resulting distribution is only _likely_ to be _approximately_ correct
- This distribution will likely be very distorted in regions that are unlikely
- It can require exponentially many samples to resolve events
- Equally likely paths are likely to be close, but almost guaranteed not to have exactly the same mass estimate. Worse, no sampling method will ever be able to conclude that $p(x \land y) \not> p(x \land \lnot y)$ with high probability, regardless of the number of samples.

Instead, we can interpret the code as branching into two worlds on a flip, each tracking the exact correct amount of mass. While this works for straight line code, such as the program above, any loop which could run an unbounded number of times will cause the program to run forever. At this point, we have removed the probabilistic component, and we now have a deterministic approximation which will converge to the answer we would like. This alleviates the msot pressing of our issues, but it is midly annoying that we will never terminate when evaluating a program with possibly unbounded iteration, even if the limit point is obvious. For instance, this program which repeatedly flips two coins until one of them lands tails:

```
start:
  x : bool = flip;
  y : bool = flip;
  z : bool = and x y;
  br z start end;
end:
  ret
```
which one can easily see results in the distribution:

\[
p \left(\begin{matrix}x \land y \land \lnot z \vphantom{\frac{1}{3}} \\
x \land \lnot y \land \lnot z \vphantom{\frac{1}{3}} \\
\lnot x \land y \land \lnot z \vphantom{\frac{1}{3}}  \end{matrix}\right)
 = \left(\begin{matrix} \frac{1}{3} \\\frac{1}{3} \\ \frac{1}{3}  \end{matrix}\right)
\]

The goal is to design an algorithm which soundly deals with issues like this, which exactly computes a distribution like this in polynomial time.

<!-- I was also hoping to implement the [the R2 paper](https://www.microsoft.com/en-us/research/project/r2-a-probabilistic-programming-system/), which more explicitly makes use of Metropolis-Hastings algorithm. -->


# What I Did
I built an abstract interpreter which exactly (to reiterate: neither approximately, nor probabilistically) solves for the distribution of any program with finite state space, together with tools for generating random programs for evaluation, as well as some tools for observing and looping programs.

To the language specification I have added three instructions,

 - `flip` : an instruction which stores a random boolean in its target destination
 - `obv` : a probabalistic version of `assert`: invalidates any run in which 

## Design
There are several standard solutions to doing this inference.

### [ 1 ]  Abstract Interpretation and Jacobi Iterates


### [ 2 ]  Stationary Distributions on Markov Chains
The second natural view of a probabilistic program is as a Markov chain. In a very clear way, a program describes exactly the data required to transition from one state (including both the environment variables and the program counter) to a distribution over next states. In particular, the transition $T_{i,j}$ is the probability of transitioning to state $j$ given that you're in state $i$. For a deterministic program, $T_{i,j}$ is a function, and therefore has exactly a single one in each row, and zeros elsewhere; this structure is

While probabilistic bril programs are not deterministic, they are far from being arbitrary matrices --- because they're probablistic transition matrices $A$ must have $\sum_{i} A_{i,j} = 1$ for all $i \in \mathcal S$.

flip instructions only change the state of the environment, can only fork into two states, and all . Moreover,


#### Eigenspace
Given an oracle for computing the eigenvectors of this transition matrix, the right thing to do is clear:

\[ \lim_{n \to \infty}  \mathbf A^n \vec s  = \lim_{n \to \infty}  \mathbf A^n \vec s \]



## Implementation

```
[ 2, Map { 'one' => 1n, 'y' => 24n, 'x' => true } ] 5.960464477539063e-8
[ 6, Map { 'one' => 1n, 'y' => 24n, 'x' => false } ] 5.960464477539063e-8
[ 'done', Map { 'one' => 1n, 'y' => 23n, 'x' => false } ] 1.1920928955078125e-7
[ 'done', Map { 'one' => 1n, 'y' => 22n, 'x' => false } ] 2.384185791015625e-7
[ 'done', Map { 'one' => 1n, 'y' => 21n, 'x' => false } ] 4.76837158203125e-7
[ 'done', Map { 'one' => 1n, 'y' => 20n, 'x' => false } ] 9.5367431640625e-7
[ 'done', Map { 'one' => 1n, 'y' => 19n, 'x' => false } ] 0.0000019073486328125
[ 'done', Map { 'one' => 1n, 'y' => 18n, 'x' => false } ] 0.000003814697265625
[ 'done', Map { 'one' => 1n, 'y' => 17n, 'x' => false } ] 0.00000762939453125
[ 'done', Map { 'one' => 1n, 'y' => 16n, 'x' => false } ] 0.0000152587890625
[ 'done', Map { 'one' => 1n, 'y' => 15n, 'x' => false } ] 0.000030517578125
[ 'done', Map { 'one' => 1n, 'y' => 14n, 'x' => false } ] 0.00006103515625
[ 'done', Map { 'one' => 1n, 'y' => 13n, 'x' => false } ] 0.0001220703125
[ 'done', Map { 'one' => 1n, 'y' => 12n, 'x' => false } ] 0.000244140625
[ 'done', Map { 'one' => 1n, 'y' => 11n, 'x' => false } ] 0.00048828125
[ 'done', Map { 'one' => 1n, 'y' => 10n, 'x' => false } ] 0.0009765625
[ 'done', Map { 'one' => 1n, 'y' => 9n, 'x' => false } ] 0.001953125
[ 'done', Map { 'one' => 1n, 'y' => 8n, 'x' => false } ] 0.00390625
[ 'done', Map { 'one' => 1n, 'y' => 7n, 'x' => false } ] 0.0078125
[ 'done', Map { 'one' => 1n, 'y' => 6n, 'x' => false } ] 0.015625
[ 'done', Map { 'one' => 1n, 'y' => 5n, 'x' => false } ] 0.03125
[ 'done', Map { 'one' => 1n, 'y' => 4n, 'x' => false } ] 0.0625
[ 'done', Map { 'one' => 1n, 'y' => 3n, 'x' => false } ] 0.125
[ 'done', Map { 'one' => 1n, 'y' => 2n, 'x' => false } ] 0.25
[ 'done', Map { 'one' => 1n, 'y' => 1n, 'x' => false } ] 0.5
```


# Difficulties


# Evaluation
