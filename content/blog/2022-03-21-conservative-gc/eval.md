## Impact of Conservatism

The authors evaluated the impacts of conservativeness on collector
mechanisms and design, specifically on the number of roots tracked,
filtering, excess retention, and pinning.

The takeaways are as follows:

- *Ambiguous Pointers*: A challenge in conservative collection is how
   often a non-pointer object may be interpreted as a pointer. The
   authors found that conservative scanning results in _1.6x_
   more identified "roots" than exact scanning.

- *Excess Retention*: (Not significant probelms for conservative collectors)

- *Pointer Filtering*: The authors compare the time performances
  between their object map and the state-of-the-art BDW free-list
  introspection, which are functionally equivalent ways of filtering
  ambiguous roots. Object maps had a higher overhead in total,
  mutator, and collection times primarily due to (1) setting bits at
  allocation time, and (2) a space penalty that results from having to
  store the map. The authors concluded that in the context of a
  non-moving collector, BDW is clearly the better solution; however,
  copying allows for a greater performance benefit.

- *Pinning Granularity*: Conservative collectors need to pin the
   "referents" of ambiguous pointers, but the effect of pinning
   depends on the collector's pinning granularity. In Bartlett-style
   page pinning granularity (used by the Mostly Copying Collectors
   (MCC)), where the "referent" and all other objects on the page that
   it resides in are retained, _2.1%_ of the live heap was impacted.
   In the 256B line granularity (used by the Immix family of
   collectors), where only the "referent" is pinned, _0.2%_ of the
   live heap was impacted. Therefore, the authors concluded that
   pinning at the line granularity is significantly less impactful.

## Performance Evaluation

The authors also evaluated the performance impacts of conservatism, by
comparing newly developed conservative collection techniques (the
conservative versions of Reference Counting, Immix, Sticky Immix,
and RC Immix) against their exact counterparts and previous
state-of-the-art conservative collectors.
