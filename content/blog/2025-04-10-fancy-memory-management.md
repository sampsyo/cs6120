+++
title = "Memory Compaction via Meshing"
[extra]
latex = false
[[extra.authors]]
name = "Lisa Li"
[[extra.authors]]
name = "Bryant Park"
[[extra.authors]]
name = "Mariia Soroka"
[[extra.authors]]
name = "David Han"
+++

This blog post is about [Mesh: Compacting Memory Management for C/C++ Applications](https://dl.acm.org/doi/10.1145/3314221.3314582), published in PLDI 2019.

Many modern programming languages—like Java, Go, and Python—use automatic memory management. Their garbage-collected runtimes periodically find and reclaim unreachable memory, reducing the burden on the programmer to manually deallocate memory when it is no longer needed. This automatic management simplifies development and avoids common bugs like memory leaks and double-frees, which can occur in languages which use manual memory management. However, garbage collection comes with trade-offs. It adds runtime overhead (for instance, in the form of reference counting or object graph traversal) and can lead to unpredictable stop-the-world collection phases, which are undesirable in latency-sensitive applications. In contrast, C/C++ uses manual memory management, giving programmers direct control over allocation/deallocation via `malloc` and `free`. This fine-grained management over the layout of memory and the timing of deallocation makes C/C++ popular in performance-sensitive contexts, where deterministic behavior and tight control over resources is essential.

### Fragmentation
Any program which uses dynamic memory allocation is susceptible to fragmentation, which occurs when free memory is scattered across the heap in non-contiguous chunks. As a result of fragmentation, it is possible that the memory system is unable to satisfy a request even when the requested allocation size exists (non-contiguously) in the heap. Garbage-collected languages address this via a compaction phase—the runtime environment controls all references to objects in memory, and therefore has the ability to move live objects closer together. This consolidates free space into larger contiguous blocks, mitigating external fragmentation.

In contrast, C/C++ do not provide a mechanism to safely track or update all references to allocated memory. Pointers can be stored anywhere and manipulated freely, making it impossible for a memory manager to move objects—changing their addresses—without risking program correctness. This fundamental requirement for fixed addresses is a significant limitation of C/C++, and fragmentation remains a serious concern for long-running and memory-intensive applications. Despite existing efforts to design effective memory managers, [Robson bounds](https://dl.acm.org/doi/pdf/10.1145/321832.321846) indicate these allocators can still suffer from significant fragmentation—in the worst case, the extra memory required grows proportionally to the log of the ratio between the largest and smallest allocated object sizes.

## Summary and Contributions
[MESH](https://github.com/plasma-umass/Mesh) is a drop-in replacement for `malloc` and `free` that implements compacting memory management for C/C++. MESH exploits the virtual memory abstraction to move objects across physical pages while maintaining virtual addresses by combining randomization and a new technique called *meshing*, enabling compaction without relocation. In doing so, MESH avoids catastrophic fragmentation and breaks the Robson bounds with high probability.

Meshing is based on a mechanism introduced by the [Hound](https://github.com/plasma-umass/Hound) memory leak detector to merge physical pages with live objects at non-overlapping offsets. Virtual addresses are not affected by merge operations—instead, as a result of a merge, multiple virtual pages are mapped to the same physical page. Doing so increases the memory utilization of in-use pages while returning free pages to the OS. A key contribution of MESH is the addition of randomization to meshing in two forms—allocation and finding meshable pages—which permit strong formal guarantees on the library’s effectiveness.

### Randomized Allocation
The authors motivate the need for randomized allocation with a pathological example: programs which allocate a small amount of memory at the same offset in each page, resulting in highly underutilized yet unmeshable pages. To avoid this, MESH allocates objects for each thread based on a thread-local shuffle vector, which returns a random offset from within the associated span for each allocation. This conceptually simple strategy provides impressively high likelihood of successful meshing and circumvents the aforementioned adversarial case.

### SplitMesher
Another interesting contribution of MESH is `SplitMesher`, a procedure which approximates ideal meshing with high probability by again leveraging randomization. While solving the optimal meshing problem—which results in freeing the maximum number of pages possible—is NP-Hard in the general case, MESH takes advantage of its randomized allocator to simplify meshing to a matching problem, arguing that finding meshable cliques of size 2 is a sufficient approximation. This idea is implemented as `SplitMesher`, an algorithm which randomly probes two sets of spans to find meshable pairs between them. 

The procedure is configurable via a parameter `t` which determines the tradeoff between the runtime and quality of the meshing. Though the authors mention determining this parameter empirically to be 64, we found this explanation slightly unsatisfying, as we would have been interested in a theoretical justification (given how rigorous the rest of the paper was) or experimental results.

## Discussion

### Hardware Interaction
During discussion of this paper both asynchronously and in-person, the impact that randomized meshing would have on cache performance was a frequently raised concern. One way is the actual randomized allocation. Since objects that are allocated together are likely to be accessed together, in theory this could hurt consecutively allocated objects because they would no longer be contiguously allocated. The paper authors also addressed this first concern in a [comment](https://news.ycombinator.com/item?id=19185474) on Hacker News, saying that performance regressions were found to be caused by unoptimized allocator performance rather than ineffective caching. 

Another potential concern raised during discussion was that the actual act of compaction could pollute the data cache, as objects are copied from one physical page to another. However, since the results did not show a significant performance regression, we concluded during discussion that this would probably not be a concern in most cases. 

### Evaluation Critique
The evaluation presented in this paper drew our attention the most. The empirical results—16% memory consumption reduction in Firefox and 39% in Redis—are particularly compelling, and they make a strong case for the adoption of MESH in similar memory-intensive systems. Furthermore, MESH’s heap size reduction in Redis is equivalent to the reduction achieved by Redis’ custom defragmentation implementation, an impressive result for a drop-in library replacement.

However, there were some aspects of the empirical evaluation we did not find fully satisfying. In particular, the performance overhead of MESH is only vaguely described; in the case of Firefox the overhead is 1%, which could be considered negligible, but the custom Ruby microbenchmark displays an overhead of over 10%. Given the significant performance cost that MESH can introduce, many of us hoped to see further details on the breakdown of this overhead—understanding whether the cost is associated with certain operations, like finding meshable spans or performing the actual meshing, would better inform us in regards to the ideal use cases and tradeoffs of MESH.

Another topic which arose in our discussion was the authors’ use of the SPECint benchmark suite to evaluate MESH. Given that most of these benchmarks are not memory-intensive and do not exhibit interesting memory patterns—the authors even mention this themselves—we were curious why the suite was included at all. Benchmarks which target memory performance and efficiency existed at the time of this paper, and we felt an evaluation using these would have been more informative.

## History

This paper is fairly recent, having been published in 2019. As of now, it has 34 citations on Google Scholar, and the GitHub [repository](https://github.com/plasma-umass/Mesh) hosting the library has 1.8k stars. 

When the paper was initially published, there was some [interest](https://github.com/jemalloc/jemalloc/pull/1489) in adding a mesh implementation to jemalloc, a leading general-purpose malloc implementation that is most notably used by FreeBSD. Unfortunately though, the engineer who was adding the implementation [left the jemalloc](https://github.com/jemalloc/jemalloc/pull/1493) team a few months after the initial exploration PRs, and nobody else picked up the work. 

Mesh has also been cited by later work comparing their performance to Mesh’s performance, such as Microsoft’s [mimalloc](https://github.com/microsoft/mimalloc) and a [machine learning-based memory allocator](https://dl.acm.org/doi/abs/10.1145/3373376.3378525) that seems to have made more of a splash than Mesh itself. The learning-based memory allocator paper is actually quite interesting because it also seeks to reduce fragmentation specifically for larger pages and is claimed by the authors to reduce fragmentation on huge pages by “an order of magnitude over” Mesh. Since Mesh is highly effective at smaller page sizes and not as effective with larger pages, the authors suggest that their learning-based approach could be combined with meshing to effectively defragment memory at both smaller and larger page sizes. 

Although we all found the paper quite innovative in its use of randomized algorithms to effectively reduce fragmentation in unmodified C/C++ code, it seems from our research that Mesh has had a limited impact on the memory allocator landscape, and is yet to be implemented in any commercial or widely used malloc implementation. It seems to be most often cited as a point of comparison by later works on memory allocators, which makes sense given that it was implemented as a drop-in replacement for malloc that can be easily added to performance evaluations. We anticipate future work that builds on the concepts introduced by Mesh and incorporates it with other novel memory allocation techniques or into other languages without garbage collection, such as Zig or Rust. 
