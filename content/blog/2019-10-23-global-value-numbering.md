+++
title = "Global Value Numbering"
[extra]
bio = """
  [Alexa VanHattum][] is a second-year student interested in the intersection of compilers and formal methods. She also enjoys feminist book clubs and cooking elaborate [fish truck][] meals.

  [Gregory Yauney][] is a second-year student working on machine learning and digital humanities.
  
[alexa vanhattum]: https://www.cs.cornell.edu/~avh
[gregory yauney]: https://www.cs.cornell.edu/~gyauney
[fish truck]: https://www.triphammermarketplace.com/events/
"""
[[extra.authors]]
name = "Alexa VanHattum"
link = "https://www.cs.cornell.edu/~avh"
[[extra.authors]]
name = "Gregory Yauney"
link = "https://www.cs.cornell.edu/~gyauney"
+++


## SSA

Difficulties:
- Dominance tree: we needed the direct children, not the entire dominance relation. We ran the transitive reduction on the dominance relation.
- We only add a phi node for a variable if that variable is defined more than once.
- Whether to start with phi nodes with duplicated arguments or to add them as we processed the node. We keep track of where the phi argument sources are from.

## Global value numbering
