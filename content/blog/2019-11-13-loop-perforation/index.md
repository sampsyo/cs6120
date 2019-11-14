+++
title = "Loop Perforation"
[extra]
bio = """
  [Oliver][] is a CS PhD student in theory at Cornell, who does decision theory and category theory.

  [Alexa][] is a second-year student interested in the intersection of compilers and formal methods. She also enjoys feminist book clubs and cooking elaborate [fish truck][] meals.

  [Greg][] is a second-year student working on machine learning and digital humanities.

"""


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

## Implementation

We implemented two LLVM passes:

1. a function pass to gather information about all loops in a program

2. Perforate all loops with given rates

Both work in conjunction with

3. Python `driver.py`

## A Meandering Tour of Loop Perforation

To understand the interplay between our LLVM pass, the user-defined error metrics, and the python driver, let's consider a toy example.
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
For `sum_to_n1`, let's define a single error metric that's the ratio between our new sum answer and the correct answer:

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

Now, we can hand off this little application to the driver, to determine which loops it can successfully perforate. The driver takes an argument for what level of error is acceptable; we said we cared about the order of magnitude, so let's say the error can be 50% and we'd still be happy:

```
$ python3 driver.py tests/sum_to_n -e 0.5
```

Let's walk through what happens now. First, the driver needs a basic sense of what the correct behavior for this application should be, so it builds and executes the application (on the representative input, if provided.)
For `sum_to_n`, our basic understanding of arithmetic holds up, and we get that the sum of the numbers from 0 to 4 is indeed:

```
10
```

This output is saved to disk for later comparisons.

Executing the application a single time assumes that the application's output is deterministic, which is obviously a huge potential blindspot in loop perforation.
In particular, our implementation does nothing to detect non-determinism, which seems consistent with the prior published work on this topic.

After the driver executes the standard variant of the application, it needs to determine what loop structures the program has to exploit.
To accomplish this, the driver runs a function pass, `LoopCountPass`.
Because we are in LLVM-land, this pass gets to rely on existing LLVM infrastructure for most of the heavy lifting.
The pass invokes two dependent passes, `llvm::LoopInfo` and `llvm::LoopSimplify`, which return statistics and simplify loops to a canonical form where possible, respectively.
Our pass then examines which of these loops have both been successfully been converted to a simple form and have a canonical induction variable.
We write these loops, which we consider to be perforation candidates, out to disk as a JSON file.
For `sum_to_n`, the implementation has example one loop in a simple form, so the resulting JSON looks something like this:

```
{
    "sum_to_n-phis.ll": {
        "sum_to_n": [
            "%2,%4,%6"
        ]
    }
}
```

We use functionality from `llvm::Loop::Print()` to get the name of each loop, which includes which basic blocks are included in the loop (here, `%2`, `%4`, and `%6`) as well as their role within the loop.
Here, we enter the loop at block `%2`, then either exit the block or branch to blocks `%4`, `%6`, and back to `%2`.

Next, the driver needs to explore how far it can mangle each loop before the results become unacceptable (remember, here that means with error under 50%).
The driver iteratively perforates each candidate loop with a set of possible perforation rates---for this example, 2, 3, 5.
More concretely, the driver invokes a second LLVM pass, `LoopPerforationPass`, that finds canonical induction variables and replaces them with constants multiplied by the desired rate.
For our toy example, conceptually this means changing the loop increment expression from:

```
for (int i = 0; i < n; i++) {
    ...
}
```

To:
```
for (int i = 0; i < n; /* Perforated rate here -> */ i += 2 ) {
    ...
}
```

At the LLVM intermediate representation level, this changes this blocks' implementation from:

```
; <label>:2:
  %.01 = phi i32 [ 0, %1 ], [ %5, %6 ]
  %.0 = phi i32 [ 0, %1 ], [ %7, %6 ]
  %3 = icmp slt i32 %.0, %0
  br i1 %3, label %4, label %8

; <label>:4:
  %5 = add nsw i32 %.01, %.0
  br label %6

; <label>:6:
  %7 = add nsw i32 %.0, 1
  br label %2
```

To:

```
; <label>:2:
  %.01 = phi i32 [ 0, %1 ], [ %5, %6 ]
  %.0 = phi i32 [ 0, %1 ], [ %7, %6 ]
  %3 = icmp slt i32 %.0, %0.
  br i1 %3, label %4, label %8         ;; <- Branching condition doesn't change

; <label>:4:
  %5 = add nsw i32 %.01, %.0
  br label %6

; <label>:6:
  %7 = add nsw i32 %.0, 2              ;; <- Perforated rate here
  br label %2
```
After the driver perforates the desired loop, it executes that modified executable on the representative input.
The driver checks the return code of the execution and treats crashing programs as unacceptable.
It also measures the execution of each program in wall-clock time.
The output is saved to disk then compared with the standard, non-perforated output.
The comparison uses the application-specific `errors` module, calculating the error between the perforated and expected result for a number of user-defined metrics.
This driver repeats this for every potential perforation rate, and then write the results out as another JSON file.

For `sum_to_n`, this looks like the following (with some light post-processing for brevity):

```
{
    "{\{\"sum_to_n\": {\"%2,%4,%6\": 1}}}": {
        "return_code": 0,
        "time": 0.0195,
        "errors": {
            "error_ratio": 0.0
        }
    },
    "{\{\"sum_to_n\": {\"%2,%4,%6\": 2}}}": {
        "return_code": 0,
        "time": 0.0194,
        "errors": {
            "error_ratio": 0.4
        }
    },
    "{\{\"sum_to_n\": {\"%2,%4,%6\": 3}}}": {
        "return_code": 0,
        "time": 0.0197,
        "errors": {
            "error_ratio": 0.7
        }
    },
    "{\{\"sum_to_n\": {\"%2,%4,%6\": 5}}}": {
        "return_code": 0,
        "time": 0.0196,
        "errors": {
            "error_ratio": 1.0
        }
    }
    // ...
}
```

Here, our wall-clock execution time is essentially a wash, because the loop is so small that the difference is negligible relative to noise.
However, we can see that the error ratio does drastically change as we increase the perforation rate, from perfect for the standard run to completely unacceptable at a perforation rate of 5, as we might expect (skipping 4 in every 5 iterations is skipping every integer except 0!) Note that if our application had multiple loops, we would see a results equal to the number of loops times the number of perforation rates (one run for each.)

The final task of the driver is to combine what it learned about how much it can mangle each loop into one final answer.
That is, where we were previously perforating each loop in turn, we now want to perforate some subset of the loops that we determined don't hurt the accuracy too much.
Here, we follow the lead of the original loop perforated paper in making a greedy assumption---that we can choose an final loop perforation strategy based on joining the maximum perforation rate for each loop that that was below the error tolerance.
This strategy makes the very optimistic assumption that errors are not additive; in the published paper their system backtracks in the case that the combined executable is unacceptable.
For the purpose of this project, we simply ungracefully fail in that case.

In our `sum_to_n` toy example, this joined result sees that only a loop perforation rate of 2 produces an executable that is below our error threshold of 50%. We thus see the following final joined summary in the results:

```
    "joined_{\{\"sum_to_n\": {\"%2,%4,%6\": 2}}}": {
        "return_code": 0,
        "time": 0.020437002182006836,
        "errors": {
            "error_ratio": 0.4
        }
    }
```

To summarize, our pass took our little toy example of summing the integers up to `n` and made it much, much stupider.
We defined a simple error metric that the resulting sum must be within 50% of the real sum, which led the driver to choose a loop perforation rate of two.
The ultimate joined peroration pass thus changed the loop increment from `+= 1` to `+=2`.
This means the sum skips the loop iterations for `i = 1` and `i = 3`, resulting in a total sum `0 + 2 + 4 = 6`, which is 40% off from the correct answer of 10.
Close enough!!!!!!1! ¯\\_(ツ)_/¯

Now, an enterprising reader might have picked up on the fact that while we claim that loop perforation will make your code run _faster_, we actually executed your entire executable _way more times_ in order to find perforation rates that won't crash or completely destroy your output accuracy.
That reader would be right!
However, the promise of loop perforation (along with many other optimizations that rely on dynamic analysis) is that we can run the expensive analysis on a small, representative input, and have the performance improvements scale to much larger examples.
For this silly toy, imagine we wanted to sum a list of millions of numbers---we would not need to rerun analysis, but could simply use the same executable.

### Here's the code!

### Scope

### Design Decisions

- We directly modify the instruction that increments a loop's induction variable; Adrian implemented loop perforation differently.
- To collect loop information: decided to do a function pass instead of a loop pass or module pass:
    - the module pass is the "right way to do it" but the LoopInfo is not finished by the time this pass is run;

## Evaluation

### Tests

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



 -
