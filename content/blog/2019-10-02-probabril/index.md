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
we should be able to see represents the distributions

\[ x \]

$$ \left(\begin{array}{}
  1 & 0 \\
  0 & 1
\end{array}\right)
$$


$$ p \left(\begin{matrix}x \land y  \\
x \land \lnot y  \\
\lnot x \land y \\\lnot x \land \lnot y  \end{matrix}\right)
 = \left(\begin{matrix} .25 \\ .25 \\ .25 \\ .25 \end{matrix}\right)$$

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
which one can easily see results in the distribution

$$
p \left(\begin{matrix}x \land y \land \lnot z \vphantom{\frac{1}{3}} \\
x \land \lnot y \land \lnot z \vphantom{\frac{1}{3}} \\
\lnot x \land y \land \lnot z \vphantom{\frac{1}{3}}  \end{matrix}\right)
 = \left(\begin{matrix} \frac{1}{3} \\\frac{1}{3} \\ \frac{1}{3}  \end{matrix}\right)
$$
... will never terminate if we just split worlds on flips, because there's 

The goal is to design an algorithm which soundly deals with issues like this, which exactly computes distributions over any program with finite state space in a finite number of steps.

    
<!-- I was also hoping to implement the [the R2 paper](https://www.microsoft.com/en-us/research/project/r2-a-probabilistic-programming-system/), which more explicitly makes use of Metropolis-Hastings algorithm. -->


# What I Did
I built an abstract interpreter which exactly (to reiterate: neither approximately, nor probabilistically) solves for the distribution of any program with finite state space, together with tools for generating random programs for evaluation, as well as some tools for observing and looping programs. To the best of my knowledge, everything like this that already exists is an iterative approximation of the fixed point, rather


## Design

To the language specification I have added three instructions,

 - `flip` : an instruction which stores a random boolean in its target destination
 - `obv` : an observe primative, used for conditioning, which can be thought of as an assert --- if it fails, the world and any mass on it are destroyed, netting a sub-deistribution. If one thinks of programs as being normalized distributions (that is, conditioned on a program finishing), then this mass is re-distributed to the other runs, and this instruction is equivalent to ra restart of the program.
 - `clear` : clears the environment variables. `obv` can be compiled to a branch which restarts the program, with a `clear`. 

### Background on Exact Inference

There are at least two canonical ways of approaching this problem: one from programming languages, and one from ergodic theory. In both cases, a program $P$ can be thoguht of as a weighted graph $(\mathcal S, T)$, where the vertices 
$$ \mathcal S := \mathrm{Instructions} \times \mathrm{Env}  $$

are pairs consisting of the program counter and the environment state, and the weight $T_{s_1, s_2}$ of the edge between states $s_1$ and $s_2$ is the probability of transitioning from state $s_2$ from state $s_1$. Note that this graph is incredibly sparse, as each state can only move to one or two other states. 
 
#### [ 1 ]  Abstract Interpretation and Jacobi Iterates
The first thing we can do is in the same spirit as information flow analysis: we interpret programs abstractly --- that is, run them by keeping track of some some restricted information that necessarily must be true about each variable, rather than its exact value.  

Consider the abstract domain $\mathcal D = (2^{\Delta \mathcal S}, \subseteq, \varnothing, \bigcup)$ of sets of distributions over states, which we will call $\Delta \mathcal S$, can be endowed with a natural order $\preceq$ on wich $T$ is monotonic, making it a complete partial order (CPO), i.e., an ordered set with arbitrary superema. Because it is a CPO and $f$ is monotonic, there is a least fixed point of $x$ of $f$ such that $x \succeq s$ for any $s \in \mathcal S$, computed by

$$ x := \mathrm {lfp}^{\preceq} (f,s) =  \lim_{n \to \infty} f^{n} (s) $$

The values obtained by stopping at any given point are called the Jacobi iterates, and are the basis of Cousot style abstract interpretation. However, even if the state space $\mathcal S$ is finite, the set of distributions over them is decidedly not --- and this procedure will not terminate. In practice, to get termination people sacrifice completeness to get a sound, terminating abstract interpreter, pulling tricks such as [widening](https://en.wikipedia.org/wiki/Widening_(computer_science)#Use_in_Abstract_Interpretation).  In this setting, this 

#### [ 2 ]  Stationary Distributions on Markov Chains
The second natural view of a probabilistic program is as a Markov chain. In a very clear way, a program describes exactly the data required to transition from one state (including both the environment variables and the program counter) to a distribution over next states. In particular, the transition $T_{i,j}$ is the probability of transitioning to state $j$ given that you're in state $i$. For a deterministic program, $T_{i,j}$ is a function, and therefore has exactly a single one in each row, and zeros elsewhere; this structure is


##### Projecion into Eigenspaces
Because it is a contracting map, the Banach fixpoint theorem tells use that a fixed point exists, and it can be calculated by iteratively applying the matrix $T$ to any point in our space

Given an oracle for computing the eigenvectors of this transition matrix $\mathbf T$, the right thing to do is clear:

$$ \lim_{n \to \infty}  \mathbf T^n \vec s  = \lim_{n \to \infty}  \mathbf U \Sigma^n \mathbf V^T \vec s $$

where $\mathbf U \Sigma \mathbf V^T$ is the singular decomposition of $\mathbf T$ --- that is, $\mathbf U$ and $\mathbf V$ are unitary and $\Sigma$ is a diagonal matrix of the singular values of $\mathbf T$. Because we know $\mathbf T$ is a (sub)stochastic matrix, we know that it can have no eigenvalue greater than 1, and if the program has any chance of returning, the return statement corresponds to an eigenvector that does in fact have corresponding eigenvalue 1. It is easy to see that any singular value that is less than 1 will ultimately go to zero, and 

<!--hr/-->
### Algorithm

The key insight is that the limits point of a single cycle can be computed the moment you spot the cycle, and know where all of the "off-ramps" are. For instance, if your program looks like this:

![example-graph](graph-sketch.png)

then the moment you've seen the path $a \to b \to c$, and realized that the probability mass has dropped from $1$ to $\frac{1}{8}$ by going around the circle, we see that that we will ultiately end up with a geometric series

$$ 1 + \frac{1}{8} + \frac{1}{8^2} + \cdots  =  \frac{1}{1 - \frac{1}{8}} $$ 

That is, we can immediately pass to a limit, by removing all weight at the origin of the cycle, and multiplying the probability masses which branch off by $\frac{8}{7}$, which means the resulting graph looks like this:


In any case, this results in the following algorithm:

```javascript
let x = $a$.
```

#### How is this possible?

While probabilistic bril programs are not deterministic, they are far from being arbitrary matrices --- because they're probablistic transition matrices $A$ must have $\sum_{i} A_{i,j} = 1$ for all $i \in \mathcal S$.

flip instructions only change the state of the environment, can only fork into two states, and all . Moreover,


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
