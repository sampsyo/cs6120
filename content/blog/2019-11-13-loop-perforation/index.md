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

## Overview

## Evaluation

### Feature Wish list:
- criticality testing
- rates for nested loops: possibly exponential rates moving inward
- accelerated loop perforation
- storing previous values instead of skipping
- Outrageous baselines that work as well, e.g. skipping random instructions
- the loss of accuracy may introduce bias in, e.g., ML
- parsec
- accuracy measure
- should we bother with their greedy exploration?


## Implementation

 - There is a function pass that gets information about the loops out to python. This is run by calling `opt` with the flag `-loop-count`.
    - We collect json information about all loops (including the funciton, module, whether or not there's an induction variable...)
    - in the destructor, we save the information that ended up in each module to a json file of the same name.


## Difficulties

 - To collect loop information: decided to do a function pass instead of a loop pass or module pass:
    - the module pass is the "right way to do it" but the LoopInfo is not finished by the time this pass is run;

 -
