+++
title = "Fast Conservative Garbage Collection"
[[extra.authors]]
name = "Ayaka Yorihiro"
link = "https://ayakayorihiro.github.io"
[[extra.authors]]
name = "Shubham Chaudhary"
link = "https://www.cs.cornell.edu/~shubham/"
+++

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
twofold:

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


# Evaluation

