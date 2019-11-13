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
