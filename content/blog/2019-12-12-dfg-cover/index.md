+++
title = "Finding Redundant Structures in Data Flow Graphs"
[extra]
bio = """
  [Oliver][] is a CS PhD student in theory at Cornell, who does decision theory and category theory.

  [Alexa][] is a second-year student interested in the intersection of compilers and formal methods. She also enjoys feminist book clubs and cooking elaborate [fish truck][] meals.

  [Greg][] is a second-year student working on machine learning and digital humanities.

[alexa]: https://www.cs.cornell.edu/~avh
[greg]: https://www.cs.cornell.edu/~gyauney
[oliver]: https://www.cs.cornell.edu/~oli
[fish truck]: https://www.triphammermarketplace.com/events/
"""
latex = true

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

## Go with the flow

Fun introduction!

## Data flow graphs for computational acceleration

- Dependencies matter!
- DFGs nicely model spatial acceleration

## Building data flow graphs from LLVM

- Trade-offs:
- Machine instructions vs. IR instructions
- Static vs. dynamic DFGs
- Getting simple data flow "for free" vs. complexities of control flow

## Matching fixed DFG stencils

- Defining node matches
- Finding isomorphisms

## Generating common DFG stencils

- n-node vs. n-edge stencils
- Beam search
- Scaling

## Static and dynamic coverage

- Annotations on LLVM
- Embench benchmark suite
- Use fast stencils for slow/big applications

## Ongoing directions
- Extend to hyperblock/superblock
- Compare against dynamic DFGs
- Evaluate on accelerated hardware
- Find stencils for groups of applications
