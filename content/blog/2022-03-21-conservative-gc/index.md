+++
title = "Fast Conservative Garbage Collection"
[[extra.authors]]
name = "Ayaka Yorihiro"
link = "https://ayakayorihiro.github.io"
[[extra.authors]]
name = "Shubham Chaudhary"
link = "https://www.cs.cornell.edu/~shubham/"
+++

# Fast Conservative Garbage Collection

This paper explores the common challenges of implementing a efficient
conservative garbage collectors in managed languages, namely excess
retention and pinning caused by ambiguous references. Additionally,
the authors introduce the concept of an optimized object map, which
tracks alive objects, which mitigates pinning.

# Background

Within language implementations, memory management can be done by
either _exact_ garbage collectors, which are fully aware of all
references, or _conservative_ garbage collectors, which must deal with
_ambiguous references_ that could be either pointers or values.

Becuase exact collectors know references, they can move objects and
filter out more dead references. However, exact collectors impose
nontrivial engineering and performance challenges, such as having to
maintain a shadow stack. In many situations, it is also impossible to
implement an exact collector. So, work on implementing effective and
efficient conservative collectors is necessary.

The key challenges in implementing conservative collectors are
as follows:

1. Excess retention: Because an ambiguous reference _could_ point to
an object, that "referent" cannot be collected. This is because the
ambiguous reference _could_ be a valid pointer to that object. So,
actually dead objects and their transitively reachable descendants
will be kept alive, causing extra space to be taken.

2. Pinning: Because an ambiguous reference _could_ be a value, the
collector can't modify it. So, in the case that the ambiguous
reference is actually a pointer, the referents cannot be moved,
causing fragmentation.

# Contributions

The main contributions of this paper include:

- A detailed examination on conservative garbage collection, including
  the first detailed study of impacts of exact and conservative
  collectors in practice.

- The idea of introducing an object map that precisely tracks alive
  objects to filter ambiguous roots.

- The design, implementation, and evaluation of new conservative
  collectors that use the object map, against their prior conservative
  collectors.
  - Conservative RC Immix, introduced in this paper, is 1%
  _faster_ than Gen Immix, the best-performing exact collector.

# Techniques

## Existing Exact Collectors

## Making Collectors Conservative


# Evaluation

The authors conduct an extensive evaluation in two parts: (1)
understanding the impact of conservatism by comparing between exact
and conservative versions of collectors; (2) a performance evaluation
comparing the conservative versions of RC, Immix, Sticky Immix, and RC
Immix against existing state-of-the-art conservative garbage
collectors.

## Impact of Conservatism


## Performance Evaluation
