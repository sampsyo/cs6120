+++
title = "Loop Perforation"
[extra]
bio = """
  [Oliver][] is a CS PhD student in theory at Cornell, who does decision theory and category theory.

  [Alexa][] is a second-year student interested in the intersection of compilers and formal methods. She also enjoys feminist book clubs and cooking elaborate [fish truck][] meals.

  [Greg][] is a second-year student working on machine learning and digital humanities.

"""
latex = true


[alexa]: https://www.cs.cornell.edu/~avh
[greg]: https://www.cs.cornell.edu/~gyauney
[oliver]: https://www.cs.cornell.edu/~oli
[fish truck]: https://www.triphammermarketplace.com/events/
"""
[[extra.authors]]
name = "Oliver Richardson"
link = "https://www.cs.cornell.edu/~oli"
[[extra.authors]]
name = "Alexa VanHattum"
link = "https://www.cs.cornell.edu/~avh"
[[extra.authors]]
name = "Gregory Yauney"
link = "https://www.cs.cornell.edu/~gyauney"
+++

## Introduction

### Here's the code!

### Scope

## Implementation

We implemented two LLVM passes:

1. a function pass to gather information about all loops in a program

2. Perforate all loops with given rates

Both work in conjunction with

3. Python `driver.py`

### Design Decisions

- We directly modify the instruction that increments a loop's induction variable; Adrian implemented loop perforation differently.
- To collect loop information: decided to do a function pass instead of a loop pass or module pass:
    - the module pass is the "right way to do it" but the LoopInfo is not finished by the time this pass is run;

## Evaluation

### Error Metrics

The [original loop perforation paper][paper] uses the following accuracy metric:

\[ \text{acc} = \frac{1}{m} \sum_{i=1}^m w_i \left|\frac{o_i - \hat o_i}{o_i}\right| \]

That is to say, it comes with a pre-selected division of the accuracy into pre-selected "components" $o_i$. Though these components are sold as a modular feature of the approach, the equation above makes it abundantly clear that each $o_i$ must be $\mathbb R$-valued, which makes the choice rather restrictive. For instance, this means that matrix and vector accuracy calculations **must be** weighted sums of their dimensions. Moreover, overwhelmingly there is no good choice for one component to be weighted over another: the representation is forced by the restriction to real valued outputs of programs, and so anything encoded across multiple components cannot be re-weighted.

This means that, of the common accuracy metrics used for images, matrices, etc., only the $l_1$ loss can be encoded

Other metrics

 -**L2 Loss**
 -

### Tests

To understand the interplay between our LLVM pass and the python driver, let's consider a toy example.
Say we want to write a silly function that sums the integers from 0 to some number `n`:

```
int sum_to_n(int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
       sum += i;
    }
    return sum;
}

int main(int argc, char const *argv[]) {
    printf("%d\n", sum_to_n(5));
}
```

You can imagine that perforating the loop here is a _pretty_ bad idea: this implementation doesn't have a ton of wiggle room in getting a totally correct result.
However, if we suspend disbelief for a moment and imagine that some poor soul only cares about the _order of magnitude_ of the resulting sum, then perforating this loop becomes an interesting task.

Conceptually, the driver just needs to take in the `sum_to_n` implementation and some oracle that can answer the question: "is this perforated implementation _good enough_?"
(For more complicated applications, the driver also needs a representative input, but for this example the executable takes no arguments.)
So, let's also tell the driver how wrong the ultimate answer can be.

To do this, we require a python implementation of an `errors` module.
At a high level, `errors` should tell the driver 1) what error metrics we care about for this application, and 2) a float value between 0 and 1 for each metric (0 is perfect, 1 is unacceptable.)
For `sum-to-n1`, let's define a single error metric that's the ratio between our new sum answer and the correct answer:

```
# Provide the name of each metric we care about
error_names = ["error_ratio"]

# The arguments `standard_fn` and `perforated_fn` are filenames of output files
def error(standard_fn, perforated_fn):
    standard = int(get_contents(standard_fn))
    perforated = int(get_contents(perforated_fn))

    delta = abs(standard - perforated)
    ratio = delta / standard

    return {"error_ratio" : ratio}
```

Now, we can hand off this little application to the driver, to determine which loops it can successfully perforate:

```
$ python3 driver.py tests/sum-to-n
```

Let's walk through what happens now.
### Benchmarks from PARSEC




### Feature Wish list:
- criticality testing
- accelerated loop perforation
- storing previous values instead of skipping
- Outrageous baselines that work as well, e.g. skipping random instructions
- the loss of accuracy may introduce bias in, e.g., ML
- call adrian's pass to compare
- rates for nested loops: possibly exponential rates moving inward
- fit to one input, test on others.

#### todo
- run on all represenatitve inputs
- plot speedups
- fix matrix errors (same size)
- with some fixed error, graph: perforated vs standard

#### finished
- do search and return the best
- take out condition we can't perforate main
- get benchmarks working
- parsec
- should we bother with their greedy exploration?
- accuracy measure


## Implementation

 - There is a function pass that gets information about the loops out to python. This is run by calling `opt` with the flag `-loop-count`.
    - We collect json information about all loops (including the funciton, module, whether or not there's an induction variable...)
    - in the destructor, we save the information that ended up in each module to a json file of the same name.


## Difficulties



[paper]: https://dl.acm.org/citation.cfm?id=2025133
