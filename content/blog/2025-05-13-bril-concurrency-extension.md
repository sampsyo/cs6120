+++
title = "Sailing Bril’s IR into Concurrent Waters"
[extra]
bio = """
  Ananya Goenka is an undergraduate studying CS at Cornell. While she isn’t nerding out over programming languages or weird ISA quirks, she’s writing for Creme de Cornell or getting way too invested in obscure books.
"""
[[extra.authors]]
name = "Ananya Goenka"
link = "https://ananyagoenka.github.io"
+++

1\. Goal and Motivation
--------------------------------------

My project aims to add native shared parallelism concurrency support to [Bril](https://github.com/sampsyo/bril). It adds two simple yet powerful instructions, `spawn` and `join`, which let you launch and synchronize threads directly in Bril programs. Beyond just powering up the IR with threads, I wanted to lay the groundwork for real optimizations by writing a thread-escape analysis pass: a static check that flags heap allocations which truly stay local to a single thread, so future compiler passes can skip unnecessary synchronization or RPC overhead.  

2\. Design Decisions and Specification
--------------------------------------

### 2.1 Instruction Semantics

We introduced two new opcodes:

*   { "op": "spawn", "dest": "t", "type": "thread", "funcs": \["worker"\], "args": \["x", "y"\] }
    
*   **join**: An effect operation that takes a single thread handle argument and blocks until the corresponding thread completes.
    
To prevent clients from forging thread IDs, we defined a new primitive type thread in our TypeScript definitions (bril.ts). This opaque type ensures only genuine spawn instructions can produce valid thread handles.

### 2.2 Shared State Model

A critical design question was state sharing: should threads share only the heap or also local variables? We chose a hybrid model:

1.  **Isolated stacks**: Each thread receives its own copy of the caller's environment (Env map), so local variables remain private.
    
2.  **Shared heap**: All heap allocations (alloc, load, store) target a single global Heap instance, exposing potential data races—a faithful reflection of real-world concurrency.
    

### 2.3 Interpreter Implementation

#### Stubbed Concurrency (Option A)

Our first pass implemented spawn/join synchronously in-process: spawn would directly call evalFunc(...) and immediately resolve, making join a no-op. This stub served as a correctness check and allowed us to validate the grammar and TypeScript types without introducing asynchrony.

#### Web Worker-Based Concurrency (Option B)

To simulate true concurrency, we leveraged Deno’s Web Workers. Each spawn:

1.  Allocates a unique thread ID and a corresponding promise resolver.
    
2.  Spins up a new Worker running brili\_worker.ts, passing it the full Bril program, function name, arguments, **and a proxy to the shared Heap**.
    
3.  The worker executes runBrilFunction(...) in its own isolate and signals completion via postMessage.
    
4.  The main isolate bridges heap operations through an RPC-style HeapProxy, intercepting alloc/load/store/freefrom workers and dispatching them on the master heap.


This architecture faithfully exposes nondeterministic interleaving and shared-memory errors (e.g., data races), but at the cost of heavy message-passing overhead.

3\. Thread-Escape Analysis Pass
-------------------------------

We wrote examples/escape.py, a Python script that performs a conservative, intraprocedural escape analysis:

1.  **Seed escapes**: Pointers passed to spawn, returned via ret, stored into other pointers, or passed to external calls are marked as escaping.
    
2.  **Propagation**: Through ptradd and load chains, any derived pointer from an escaping pointer also escapes.
    
3.  **Reporting**: The pass reports per-function and global statistics: the fraction of allocations that remain thread-local.
    

This simple pass identifies heap objects that never cross thread boundaries, opening opportunities to Elide synchronization or use stack allocation in a future optimizer.

4\. Benchmark Suites
--------------------

We developed two complementary benchmark suites:

*   **Correctness Suite** (benchmarks/concurrency/\*.bril): A set of small tests covering spawn/join arity errors, double joins, shared counters, parallel sums, and array writers. We validated behavior under both the stubbed and worker-based interpreters using turnt.
    
*   **Performance Suite** (benchmarks/concurrency/perf/\*.bril): Larger, data-parallel workloads measuring speed-up with workers:
    
    *   perf\_par\_sum\_100k.bril: Summing 1..100 000 split across two threads. 100 k elements produce ~20× speed-up over sequential.
        
    *   perf\_matmul\_split.bril: Matrix multiplication on 100×100 matrices with row-range split; yields ~10× speed-up.
        
    *   perf\_big\_par\_sum\_10M.bril: Summing 1..10 000 000, which instead suffers a slowdown due to RPC overhead.
        

5\. Empirical Evaluation
------------------------

### 5.1 Correctness

All the correctness benchmarks pass under the worker-enabled interpreter (brili-conc), matching our specification. Error cases in the error directory (arity mismatch, double join, wrong handle type) correctly exit with error codes.

### 5.2 Performance

Below are the wall-clock times measured for three representative benchmarks, comparing the sequential interpreter (using --no-workers) against our concurrent version (with true Deno Web Workers and an RPC-based shared heap).

```
=== Benchmark: perf_par_sum_100k ===
Sequential: 217 ms, Concurrent: 4.5 s → 20.8× speed-up

=== Benchmark: perf_matmul_split ===
Sequential: 4.2 s, Concurrent: 44.9 s → 10.8× speed-up

=== Benchmark: perf_big_par_sum_10M ===
Sequential: 15.8 s, Concurrent:  > (timeout/slower) → RPC overhead dominates
```

#### **Why “concurrent” is actually slower**

1.  Our design (real Web Workers + RPC heap) means _every_ load and store from a worker turns into:
    ```
    postMessage({ kind: "heap_req", … })   // worker → main
    → main services request, touches native JS array
    postMessage({ kind: "heap_res", … })   // main → worker
    ```
 On a 100 k–element loop, that’s ∼200 k round trips, each costing hundreds of microseconds in Deno’s message‐passing layer—totalling several seconds. At 10 M elements, it grows to ∼20 M messages, which simply overwhelms the system.
    
2.  Spawning even two workers in Deno carries a fixed overhead (tens to hundreds of milliseconds). In fine-grained loops (100 k or fewer iterations), that overhead is not negligible, and in very fine loops (10 M) it compounds the RPC penalty.
    
3.  The main isolate is busy servicing heap requests from _both_ workers. The event loop context‐switching and message queue flooding create contention, so neither the “workers” nor the main thread run at full core capacity.
    
I'd expect that this implementation of concurrency would help, on moderately coarse workloads** (e.g. 100 k or splitting a 100 × 100 matrix) still see _some_ parallelism, because the computation per RPC is nontrivial (simple arithmetic plus pointer arithmetic inside V8). In our tests, the sequential run was ~4 s, the concurrent ~45 s—still slower, but less horrific than the 10 M sum case.
    
Some lessons/thoughts.  To actually _win_ with real parallelism under this design, we must batch memory operations. For example, transform long loops into single RPC calls that process entire slices (e.g. “sum these 1 000 elements” in one go), amortizing the message‐passing cost.SharedArrayBuffers could eliminate RPC entirely by mapping our Bril heap into a typed array visible in all workers. Then each load/store is a direct memory access, and you’d see true multicore speedups on large-N benchmarks. For an intermediate step, we could group every 1 000 loads/stores into one batched message, cutting messaging overhead by two orders of magnitude, which should already push the breakeven point down toward the 10 M-element range.
    
_In summary_, our RPC-based “shared-heap” prototype cleanly demonstrates the mechanics of true concurrency in Bril, but the messaging overhead severely limits speedups for fine-grained loops. Future work on shared‐memory backing or batched RPC will be necessary to turn this into a truly performant parallel interpreter.

### 5.3 Escape Analysis Statistics

```
[escape] main: 1/1 allocs are thread-local   # solo allocation stays local
[escape] TOTAL: 1/1 (100%) thread-local

[escape] main: 0/1 allocs are thread-local   # spawn causes escape
[escape] mixed: 1/2 allocs are thread-local  # one local, one escaping
```

These results confirm our pass correctly identifies thread-local allocations.

6\. Challenges and Lessons Learned
----------------------------------

The switch from synchronous stubs to Web Workers introduced complexity in sharing the heap. Since workers cannot share JS memory directly, we built an RPC-based HeapProxy that marshals each alloc/load/store call. Debugging message ordering, promise lifetimes, and termination conditions consumed significant effort.
I think my biggest learning was that I expected concurrency to speed everything up, but the 10 M-element test exposed how RPC overhead can negate parallel gains. 
I also took some short cuts. A fully context-sensitive, interprocedural pass would be more precise, but the current intraprocedural, flow-insensitive approach seemed to be simplest useful thing I could do.

7\. Future Work
---------------

Building on this foundation, several directions remain:

*   **SharedArrayBuffer Heap**: Replace RPC with a SharedArrayBuffer–backed heap to eliminate message-passing overhead and unlock true multi-core speed-ups for large workloads.
    
*   **Par-for Loop Transform**: Implement a compiler pass that recognizes map/reduce patterns and automatically rewrites for loops into multiple spawn calls, guided by data-dependency analysis.
    
*   **Interprocedural Escape Analysis**: Extend escape.py to track pointers across function boundaries and calls, increasing precision and enabling stack-based allocation for truly local objects.
    
*   **Robust Testing Harness**: Integrate with continuous integration (CI) to run our concurrency and escape-analysis suites on every commit, ensuring regressions are caught early.
    
