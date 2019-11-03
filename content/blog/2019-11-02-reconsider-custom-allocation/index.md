+++
title = "Reconsidering Custom Memory Allocation"

[extra]
bio = """
  [Oliver Richardson](https://cs.cornell.edu/~oli) is a second year PhD student interested in
"""
[[extra.authors]]
name = "Oliver Richardson"
link = "https://cs.cornell.edu/~oli"
+++


Memory is valuable, 

In 2002.
Custom memory allocators are lauded one of the optimizations

## Region-Based Memory Management


The example used in the papper is as follows:
```
createRegion(r);
x1 = regionMalloc(r,8);
x2 = regionMalloc(r,8);
x3 = regionMalloc(r,16);
x4 = regionMalloc(r,8);
```

## Reaps.

The authors introduce a hybrid data-structure, called a "reap": a mix between a region and a heap.


[paper]: https://dl.acm.org/citation.cfm?id=582421
[custom-alloc-blog]: https://github.com/mtrebi/memory-allocators


<!--- THOUGHTS.

* This is really an empirical claim: that people do not write or maintain their custom memory allocators properly. Obviously, antything customized for your domain can outperform what you want


Results & Evaluation:
* Benchmarks: only 8 programs. Strong incentive to chose benchmarks where the result is stronger. Worse still: there is a meta selection bias: a person who mostly interacts with benchmarks where this is a problem is more likely to come up with a paper like this.
	- To solve this, you really have to do a more convincing, reresentative sample of programs, provide reasons. This is, of course, very difficult. 
* 
* Problem with results: 

--->
