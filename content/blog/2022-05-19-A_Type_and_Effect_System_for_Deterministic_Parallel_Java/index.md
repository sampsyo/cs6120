+++
title = "A Type and Effect System for Deterministic Parallel Java"
[extra]
latex = true
bio = """
  [Nikita Lazarev](https://www.nikita.tech/) is an PhD student at Cornell's ECE department;
  Susan Garry is an undergraduate student at Cornell's CS department.
"""
[[extra.authors]]
name = "Nikita Lazarev"
link = "https://www.nikita.tech/"  # Links are optional.
[[extra.authors]]
name = "Susan Garry"
+++

### Introduction

Multithreading is notoriously difficult to reason about and error-prone. This happens due to potential data races that can be rised when two threads try to access the same region in memory simultaneously, and at least one of the accesses is a write. When not properly guarded, such memory access patterns lead to nondeterministic behavior of parallel programs and inconsistent and wrong execution. Traditional approaches using locks (e.g., mutexes) can cause a deadlock and are also difficult to reason about and debug. Therefore, designing programming models that would reinforce deterministic multithreading in a lock-free fashion is of paramount importance.

In the paper ["A Type and Effect System for Deterministic Parallel Java"](http://dpj.cs.illinois.edu/DPJ/Publications_files/DPJ-OOPSLA-2009.pdf), the authors propose making concurrency in Java deterministic while still being lock-free by using a type and effect system that involves partitioning and annotating Java programs. They introduce novel mechanisms that expand on the capabilities of previous type and effect systems, allowing for concurrent operations on recursive data structures via **region nesting**, concurrent array operations via **index-parameterized arrays** and **subarrays**, and concurrent writes to the same location in memory via **commutativity annotations**. They develop a new extension for Java called DPJ (Deterministic Parallel Java) which additionally comes with the formal verification functionality. The paper demonstrates how DPJ can be used to implement realistic parallel algorithms and verify their determinism.

### Basic Idea

The paper introduces a set of annotations for the parts of parallel programs with the aforementioned unsafe memory accesses to guard their deterministic execution at compile time. The type system defines two main classes of annotations:
* **regions** - the programmer names portions of the heap and assigns each object and variable to a region of the heap;
* **effect annotations** - denotes what regions a method reads and writes from, e.g., `reads <region-list> writes <region-list>`, where `<region-list>` is a list of regions as defined above.

The code below shows an example of a class definition (a Tree class for a tree data structure) with annotations. The example is taken from the original paper, and we added more comments in it to show the functionality.

```
// take the "global" region P for this class as a template argument.
class TreeNode<region P> {
     region Links, L, R;  // create DPJ regions
     // annotate that the member variable `mass` is in region P.
     double mass in P ;
     // annotate that member variables `left` and `right` are in region `Links`.
     TreeNode<L> left in Links;
     TreeNode<R> right in Links ;
     // Annotate effect for the function `setMass` saying that it writes to region P.
     void setMass(double mass) writes P { this.mass = mass; }
     void initTree(double mass) {
       cobegin {
         // overal effect: reads Links writes L
         left.mass = mass;
         // overal effect: reads Links writes R
         right.mass = mass;
       }
    }
 }
```

Here, the class `TreeNode` implements a simple node of a binary tree with two pointers to the `left` and `right` children and the data value `mass`. The class also defined three memory regions: the `Links` for the pointers to the children, the `L`, and the `R` for the children objects. Note that the latter annotations are recursively propagated down to the children via the parameterization region `P`. This results, for example, in different region annotations for the data value `mass` in different children nodes. This can be used to check that concurrent writes (with the `setMass()` method also annotated with the effect annotation) into the `mass` field of different nodes is safe and does not cause nondeterminism.

### Region Nesting

Since many datastructures are naturally recursive, DPJ comes with the functionality to express recursiveness of memory regions. This is called **region nesting**. Region nesting allows guarding more complex memory structures which are generated/accessed by recursion. For example, the annotations in the listing above can only guard one parent node and two children nodes, and it will not work when the tree is of an arbitrary depth. The code below shows how the program can be extended with the region nesting to fix this.

```
// take the "global" region P for this class as a template argument.
class TreeNode<region P> {
     region Links, L, R;  // create DPJ regions
     // annotate that the member variable `mass` is in region P.
     double mass in P:M ; // add P:M showing nesting of M under P
     // annotate that member variables `left` and `right` are in region `Links`.
     TreeNode<P:L> left in Links; // add P:L showing nesting of L under P
     TreeNode<P:R> right in Links ; // add P:R showing nesting of R under P
     // Annotate effect for the function `setMass` saying that it writes to region P.
     void setMass(double mass) writes P { this.mass = mass; }
     void initTree(double mass) {
       cobegin {
         // overal effect: reads Links writes L
         left.mass = mass;
         // overal effect: reads Links writes R
         right.mass = mass;
       }
    }
 }
```

The above code results in the following recursive region nesting as shown in the figure below. The key idea in this example is that the regions are also recoursive themselves, so there are the data they guard (e.g., the `mass` field). Similarly to the first example, each `mass` filed is parametrized by a different region, and this process may be repeated for an arbitrary depth of the recursion.

<p align="center">
<img src="figure_1.jpg" alt="alt_text" title="image_tooltip" width="500" />
</p>

### Subtyping

The DPJ type system supports subtyping to better express nested regions. Subtyping defines a set of rules according to which the data (objects, member variables, methods, etc.) parametrized by different regions can be cast/assigned to each other. For example, in the code above, it is not possible to write a general type of reference to point to any arbitrary node object (as each node object is strictly parametrized). In order to address this limitation, the authors define partially specified regions (denoted by `*`) and also define a set of rules on how different regions specified both fully and partially can interact with each other. A summary of the subtyping rules is shown below:

* Two regions are nested within each other, $R_1 \leq R_2$,  iff $R_1$ is contained within $R_2$, e.g., $P:M \leq P$;
* $C<R_1>$ is a subtype of $C<R_2>$ iff every region of $R_1$ is in $R_2$ ($R_1 \subseteq R_2$); examples:
  * $C<P:M>$ is a subtype of $C<P:\*>$
  * $C<P:L:*:M>$ is a subtype of $C<P:\*:M>$
  * $C<P:M>$ is not a subtype of $C\<P\>$
* B<r> is a subtype of A<r> iff  B<region R> extends A<R> {} is sound, but B<r> is not a subtype of Object

### Supporting Concurrency in Arrays

In order to allow safe concurrency in arrays, the authors define two novel capabilities: (1) **index-parameterized arrays** and (2) **subarrays**.

##### Index-Parameterized Arrays

Index-Parameterized Arrays allow to describe concurrent operations on different elements of an array. The problem seems to be trivial for arrays of primitive types (one just needs to make sure that each concurrent iteration accesses a distinct element at a time). However, it is harder for arrays of **mutable references** as here, one also needs to prove that distinct references eventually (through a chain of references) lead to distinct memory locations as well, and not a single location.

The idea behind index-parameterized arrays is the following: we need to parameterize each object reference in the array by the array index's integer expression. For example, the aforementioned `TreeNode` class should be parameterized as follows `TreeNode<Root:[10]>` for the element `10` in the array. The compiler then will check (with a symbolic integer expression solver) that all array accesses in the program going to the elements parameterized by expression `[10]` in this example, happen through the array element `10`. Similarly, to maintain soundness when references can be shuffled at the runtime, we need to make sure that a cell, say `A[i]` (where `i` is an index in the array) or any object parametrized by this `[i]` (i.e., nested objects in the reference chain) never starts to point to any object parametrized by an integer expression that evaluates to some other index of the array. If this holds, we get formally proved safe concurrent access to all the distinct elements in the array, even with mutable references.

##### Subarrays

In addition to the aforementioned Index-Parameterized Arrays, DPJ also provides the functionality to express concurrent operations on sub-arrays. This is very useful for divide-and-conquer algorithms. Here, each array can be partitioned with the new `DPJPartition` abstraction that represents a distinct memory region. `DPJPartition`s are parameterized by two integer expressions: the starting position and the length of the partition. Similarly to the above Index-Parameterized Arrays, compiler checks that all distinct concurrent memory accesses go to different partitions.

### Commutativity Annotations

This is a simple feature that attempts to generalize `read`/`write` effects by explicitly specifying what methods are commutative and therefore can be called concurrently. For example, 
```
class IntSet<region P> {
  void add(int x) writes P { ... }
  add commuteswith add;
}
```
defines that two invocations of `add` are atomic and either order of invocations produces the same result. The commutativity property is not getting checked by the compiler and should be verified independently with formal analysis.

### Evaluation

The authors evaluate their type and effect system on a set of benchmarks and show that it indeed allows expressing concurrency that provides near-ideal linear scaling of performance with the number of executors. Some algorithms (such as the Monte Carlo financial simulation) do not scale linearly due to certain limitations of DPJ such as impossibility of re-shuffling for array elements annotated with DPJ. This requires making separate copies of the arrays for parallel operations, which comes with the overhead of sequential execution. This is one of the main disadvantages of the proposed technique. Another disadvantage is the amount of developer efforts required to annotate programs. While the paper argues that the annotations themselves are very light, it seems they do require quite a complex manual analysis of the programs to put them right, and the annotations must be consistent everywhere in the program, even if the actual parallel memory access pattern is relatively simple.

Other than that, the evaluation shows that DPJ does not cause any runtime overhead in comparison with the native Java implementation, while it comes with much stronger guarantees for determinism.
