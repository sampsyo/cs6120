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



# Background on Custom Memory Allocation

Memory is necessary, but creating and freeing it takes time, and the general purpose memory management offered by C may not be optimal for your purposes. You probably know much more about the memory patterns of your application than the designers of the default memory allocator, and might be able to exploit this to save resources. This is why people write custom memory allocators. 
<!--In C, this amounts to overloading the `new` and `delete` operators. -->


#### Saving Time with Custom Allocation

For instance, suppose you have a large number of objects which you know you will evenetually create, and have some upper bound on how many you'll eventually get; moreover, you know that there are relatively few objects that are freed during the first pharse phase of the program, which takes the most time. By the second phase, which you know executes quickly, you don't need any of the objects anymore. Now, a custom memory allocator can:
 
 * Reduce the allocation time, by using only a single call to malloc, and getting enough space for them all at once (even though you might not know the exact number, or use them all at the same time, and you can't create them all yet). 
 * Save time with free calls: you can free them all at once at a the end, rather than freeing each object individually (which could still happen regularly) during phase one of the program when all of its references go away.

Allocating memory in this way would in fact be creating a _region_ allocator: you allocate one large chunk of memory at first, and then increment a pointer into it; the entire region must then be freed all at one time. 



#### Saving Space with Custom Allocation

The example above actually uses more memory than a heap. That is more common, but they can also be used to reduce the memory footprint, primarily by preventing fragmentation. For instance, if memory fragmentation is a huge issue, but you know rougly what fraction of your memory is used by objects of different sizes, you can partition it into pieces, and deal with each size class separately, which offers much better guarantees about the worst and average case fragemtation. This is commonly done in practice; [Mesh][], for instance, requires that all pages in question are of the same size class. [The paper we will analyze][paper] refers to allocators liek these as "per-class custom allocators".

For all of these reasons, in 2002, it was common practice and widely condisered a good idea to write a custom memory allocation for your program to improve performance. 


## The Obvious Drawbacks to Custom Allocation

Of course, even without any experiments, it is easy to see how doing this could be a mistake: the effectiveness of any of these relies on certain assumptions about the programs, and so any modifications need to be done while keeping the custom allocator in mind. Moreover, non-standard allocation schemes make it much more difficult to use existing tools to analyze your code. In more detail (paraphrased from paper):

* Accidentally calling the standard `free` on a custom-allocated object could corrupt the heap and lead to errors
* Custom memory allocation makes it impossible to use leak detection tools
* It also prevents you from using a different custom allocator to do something else that's more valuable in the future
* It keeps you from using garbage collection



# The Paper: Reconsidering Custom Memory Management

The paper at hand, [Reconsidering Custom Memory Allocation][paper], is packaged and sold as the thesis that custom memory allocators in practice do not out perform general ones. 

One of the key arguments of this paper is that standard baselines are not fair. Evidently the usual argument _in favor of_ cusom allocation in 2002 was a comparison against the win32 allocator, which is much slower than using a program's custom allocator. The first part of this paper is an evaluation against the Lea allocator (`dlmalloc`), which greatly  reduces the margin of victory for custom allocators not exploting regions. 


### The Taxonomy of Memory Allocators.
This paper uses the following taxonomy of custom memory allocators: 

* **Per-class allocators**. You build a separate heap for every object size class, and optimize each one separately for objects of this size. This is fast, simple, and interacts with C well --- but could be space inefficient if you don't know how much memory of each class size you will use.
* **Regions**.  An allocator that allocates large chunks of memory, puts smaller pieces within it, and then must free them all at once. They are fast and convenent, but use a lot of space and are arguably dangrous because dangling references keep things from being freed. This (nominally) requires programmers to re-structure code to keep references to the region, and free the entire region at once, resulting in a usage pattern that looks somewhat like this:
	```
	createRegion(r);
	x1 = regionMalloc(r,8);
	x2 = regionMalloc(r,8);
	x3 = regionMalloc(r,16);
	x4 = regionMalloc(r,8);
	```
	
* **Custom patterns**. Anything else --- for example, those that exploit stack-like patterns in memory allocation (the relevant benchmark is `197.parser`). The authors describe these as fast, but brittle.


## The Empirical Results

can be seen in this graph:

![Runtme Benchmarks](runtime-benchmarks.png "")

# Regions and Reaps
Recall that a heap exposes `malloc` and `free`; a region gives you a `malloc` and `freeAll`. The authors introduce a hybrid data-structure, called a "reap", which is sold as a generalization of the two.



# Another Look at the Evaluations


# From 2002 to 2019

The Lea allocator (Doug Lea's malloc, now referred to as `dlmalloc`) is now the default implementation of linux. The standard general purpose allocator to beat in evaluations is now [jemalloc](http://jemalloc.net/), which seems to be considerably more efficient [^1]. The existence of even better general purpose allocators in some ways strengthens the point made by the paper: there's even less to be gained by writing your own allocator.

On the other hand, custom memory allocation is far from dead. [Here](https://github.com/mtrebi/memory-allocators)'s a tutorial on how and why custom allocators are helpful, custom allocation still is seen as a potential reason to disregard projects such as [Mesh][], and the text on Emry Berger's [Heap Layers][heapl] project (that subsumes this paper) is still described as enablin custom memory allocation:

> "Heap Layers makes it easy to write high-quality custom and general-purpose memory allocators."

While the specifics about why people use custom allocators, and the benchmarks to support this, seem to have been wrong, it 

Furthermore, the more substantive part of this paper---the introduction and analysis of reaps---does not seem to have caught on, and 




<!--- THOUGHTS.



* This is really an empirical claim: that people do not write or maintain their custom memory allocators properly. Obviously, antything customized for your domain can outperform what you want


Results & Evaluation:
* Benchmarks: only 8 programs. Each was run on only 1 input! Strong incentive to chose benchmarks where the result is stronger. Worse still: there is a meta selection bias: a person who mostly interacts with benchmarks where this is a problem is more likely to come up with a paper like this.
	- To solve this, you really have to do a more convincing, reresentative sample of programs, provide reasons. This is, of course, very difficult. 
* These experiments don't translate to the modern world, but jemalloc and other 

General Thoughts
* This is part of a trend of that takes power away from programmers and puts it in the hands of those writing dev tools. This is often a good thing (e.g., writing higher level languages, libraries, etc.,) but also is a bit patronizing.
* Taxonomy is not at all crisp. Most things seem like they can be emulated with other things. Worry that there are false dichotomies being presented as storeies.
* Per-class allocators are easy to implement. 

One key thing to keep in mind is that this custom memory allocation is just another abstraction _that can be built with the default allocator_. It's built on top of the system `malloc`, and so anything you can 


--->

[^1]: https://suniphrase.wordpress.com/2015/10/27/jemalloc-vs-tcmalloc-vs-dlmalloc/

[paper]: https://dl.acm.org/citation.cfm?id=582421
[supermalloc]: http://supertech.csail.mit.edu/papers/Kuszmaul15.pdf
[Mesh]: https://arxiv.org/pdf/1902.04738.pdf
[heapl]: https://plasma.cs.umass.edu/emery/heap-layers.html
