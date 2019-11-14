+++
title = "Space Efficient Conservative Garbage Collection"
extra.author = "Yi Jiang"
extra.bio = """
  [Yi Jiang](http://www.cs.cornell.edu/~yijiang/) is a 2nd year Ph.D. student interested in computer architecture and systems.
"""
latex = true

+++

["Space efficient conservative garbage collection"](https://dl.acm.org/citation.cfm?id=155109) describes some inexpensive but useful techniques that can make **conservative garbage collectors** more space efficient.
Their techniques can reduce pointer misidentification to retain less memory during run time, and also prevent some of the excess retention due to less information about variable liveness than conventional collectors. Their methods can be easily incorporated into any garbage collecting allocator transparently to client programs.


# Pointer Misidentification and Solutions of Existing Conservative Garbage Collectors

The goal of a garbage collector is to retain as little memory as it can, subject to the constraint that all memory that will be accessed in the future must be retained. A garbage collector is called **conservative** if it can operate with minimal information about the layout of the client program's data. 

For conservative collectors, the most apparent potential source of excess memory retention is **pointer misidentification** (e.g., misidentifying integers as pointers). Here is a typical example:

- An integer variable contains the address of a valid but inaccessible object

The probability of such misidentification can increased if more of the address space is occupied by the heap.

Some *current ad hoc techniques* can help decrease pointer misidentification probablity. One method is to design an allocator to avoid allocating objects at address that are likely to collide with other data (properly position the heap in the address space), and to align pointers properly. Otherwise, all possible alignments must be considered by the collector and could result in more false pointers. For example, in the figure shown below, these two small integers (*0000 0009* and *0000 000a*) can also be viewed as a valid heap address (*0009 0000*) with the concatenation of the low order half word of the first integer and the high order half of the next.

![001](./001.png)

Currently, most compilers always guarantee adequate alignment. However, it is hard to determine the proper position of the heap. Thus, this paper introduces a much less ad hoc and more flexible technique to avoid pointer misidentification.


# Their Techniques

This paper's technique is composed of two steps: 
1. Normal garbage collection at regular intervals;
2. Invalid pointer recording.

The first step ensures that normal garbage collection takes place at regular intervals, with at least a *fast and initial one happening right after system startup* before any allocation begins. The second step trys to record the invalid pointers found during a garbage collection so that these addresses could be used to hold valid objects for later allocation. Below is a detailed explanation of the second step.


### Invalid Pointer Recording

They use a **blacklist** to keep a record of invalid pointers found during a garbage collection. Addresses in the blacklist, which could be valid object addresses afterwards, can not be allocated with new objects. The following algorithm is used to construct the blacklist.

![002.png](./002.png)

A naive marking algorithm is modified to support blacklisting. The only modification is in bold face. Whenever an invalid object address is found, the address would be added to the blacklist if it is in the vincinity of the heap. 

This scheme would mostly blacklist addresses that correspond to long-lived data values before these values become false references as they are the data that could possibly cause garbage to be retained indefinitely. One other thing to notice is that the scheme can eliminate the false references originating from statically allocated constant data scanned for roots by the collector, which is the most *troublesome*. Meanwhile, small and pointer-free objects can still be allocated at blacklisted address due to their little impact on erroneous retention.

### Implementation

They implemented variants of this approach in some versions of [PCR](https://dl.acm.org/citation.cfm?id=74862) and other [garbage collectors](https://dl.acm.org/citation.cfm?id=52202). They both conservatively scan the stacks, registers, static data and the heap.

Entire pages rather than individual addresses are blacklisted. In that way, the blacklist can be implemented as a bit array, indexed by page numbers. Hash table can be utilized for discontiguous heaps.

### Evaluation

They evaluate their techniques by running **a** program on different machines using both statically and dynamically linked versions of C library. The program allocates 200 circular linked lists containing 100 Kbytes each. And the collector would retain the entire list if any data points to any of the 100,000 addresses corresponding to objects in the list. The results of **storage retention** with and without blacklisting are shown in the following table.

![003.png](./003.png)

Several observations can be made based on the table:

1. Blacklisting is effective in nearly eliminating all accidental retention caused by garbage collector conservativism.
2. The numbers in the table are approximate as the results are not completely reproducible. This is due to the fact that the scanned part of the address space is polluted with UNIX environment variables. So they are specified as ranges.
3. If all interior pointers are considered valid, it would be difficult to allocate individual objects larger than about 100 Kbytes without violating the blacklist constraint, or requesting memory from the OS at a garbage collector specified location.
4. According to the paper, blacklist can be easily incorpoated into a garbage collecting allocator at *almost no performance cost*.

### Other Sources of Excess Retention

Usually, this is the end of a 2019 paper, or it may continue with some related work, future work and conclusion. However, it's published in **1993** and here is more stuff!

Another source of excess memory retention is due to the fact that conservative collectors usually have less information about **variable liveness** than conventional collectors. For example, a global variable may contain a valid pointer which is no longer used in the program. Due to the lack of knowledge of this information, such variable will remain in the stack even after garbage collection. 
The stack might be under unrealistically heavy use due to this problem, causing performance degradation in garbage collection.

So here are some useful techniques to help address this problem:
- Have the allocator and collector carefully clean up after themselves, clearing local variables before function exit.
- The allocator can try to clear areas sometimes in the stack beyond the most recently activated frame.

The first technique is to eliminate the possible impact of irregularly-triggered out-of-line allocation code and garbage collector, which is relatively rare. As the program may have a very regular execution, ensuring that the same stack location are always overwritten. So it pays to use this means (like writing 0s over stack frames after popping them) to maintain the regular execution.

The second technique sounds really confusing to me. If the stack frames is cleared individually as being popped, it seems that nothing needs to be done to the rest of the stack. So why struggling with the part of the stack beyond the most recently activated frame?

### Minor: Consequences of Misidentification

Actually, the involved data structure can greatly influence the individual false reference. 

For example, given a balanced binary tree, the expected number of vertices retained in a false reference is about the height of the tree. The height of a tree with *n* nodes lies *[log2(n+1) - 1, clog2(n+2) + b)*, which is usually tolerable. Queues and lazy lists could exhibit much worse behavior as they grow without bound and the whole data structure needs to be retained.

A more common problem is the construction of large strongly connected data structures, which could result in an unbounded memory leak if the structures are large enough. Take the data structures shown in the below graph as an example.

![004.png](./004.png)

Figure 3 and 4 depict two different data structures of a rectangular array, in which the vertices are linked both horizontally and vertically. The structure can be accessed by traversing a row/column. The left shows an embedded link representation. A false reference in this structure would result in the retention of a large fraction of the whole structure. The right shows an embedded link representation of the same structure, with a separate link representation (represented by ovals in the figure). Thus, at most a single row/column is affected.

Therefore, the embedded link version of some data structures would greatly help reduce storage retention and should be encouraged.

There is much more on this topic in another paper by Hans, [Bounding Space Usage of Conservative Garbage Collectors](https://www.hpl.hp.com/techreports/2001/HPL-2001-251.pdf) if you are interested in it.

# Conclusion and Thoughts

This paper introduces some simple but effective techniques for reducing storage retention in conservative garbage collectors. 

- I think this paper gives a thorough explanation in the pointer misidentification problem in conservative garbage collectors, as well as the current and their proposed solutions. The blacklisting solution is simple and effective, and can be easily incorporated into current collectors. 
- However, it doesn't give a thorough evaluation of their techniques. Only a handwritten program is tested for the effectiveness and no experiment is conducted on the performance cost when incorporating it into current garbage collectors. Also, they don't consider the possible memory fragmentation problem in the evaluation.

# Questions
- Why is it important to run a initial garbage collection right after system startup? Why false references originating from statically allocated constant data is the most troublesome?
- How much could fragmentation due to blacklisting influence the performance?
- Still ad hoc? Any more insightful ideas recently?
- How do you like the organization/design of the evaluation? (I thought it quite inadequate but remember it's from **1993**:)
