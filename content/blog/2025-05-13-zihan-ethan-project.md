+++
title = "Final Project: Parallel Dataflow Analysis"
[extra]
bio = """
  Ethan Uppal is a CS student at Cornell University
  Zihan Li is a CS master at Cornell University, who usually listens to music when not doing CS 
"""
[[extra.authors]]
name = "Ethan Uppal"

[[extra.authors]]
name = "Zihan Li"
+++
## Background

Dataflow analysis with the worklist algorithm can be a bottleneck for compilation speed, especially for JITs.

(The worklist algorithm is reproduced below for reference.)
```
in[entry] = init
out[*] = init
worklist = all blocks
while worklist is not empty:
	b = pick any block from worklist
	in[b] = merge(out[p] for every predecessor p of b)
	out[b] = transfer(b, in[b])
	if out[b] changed:
		worklist += successors of b
```

In our [project](https://github.com/zihan0822/para-dflow), we built a parallel dataflow solver in Rust with bitset optimizations for our flattened Bril IR.
We parallelized the KILL and GEN set computation and the condensed CFG traversal process.
We focused on (and only tested) one forward analysis (reaching-definitions) and one backward analysis (liveness).

## Preparations

To establish groundwork for the project, we coded infrastructure and tooling.
Our [Bril fuzzer](https://github.com/zihan0822/para-dflow/tree/main/bril-fuzzer), [flattened Bril representation](https://github.com/zihan0822/para-dflow/tree/main/bril) and [the parallel solver](https://github.com/zihan0822/para-dflow/tree/main/bril-analysis) are all open sourced on [GitHub](https://github.com/zihan0822/para-dflow/tree/main).
We will now discuss them in more detail.

#### Flattened Bril Representation

We implemented a flattened Bril representation to avoid the convenience grievances with the "official" representation, which, as a side effect, may have given us heap fragmentation (although performance was not the goal).
Here are some design choices we have made:
```rust
pub enum Instruction {
    Add(Variable, Variable, Variable),
    …
    Br(Variable, LabelIdx, LabelIdx),
    Call(Option<Variable>, FunctionIdx, Box<[Variable]>),
}
pub struct Function<'a> {
    pub instructions: &'a [Instruction],
    /// sorted in ascending order by offset
    pub labels: Vec<Label<'a>>,
    …
}
pub struct Program {
    pub instructions: Vec<Instruction>,
    functions: Vec<FunctionInternal>,
    strings: Vec<String>,
    labels: Vec<(usize, StringIdx)>,
}
```

This flattened representation is much easier to work with and economical to code in.
By already providing unique IDs to different things like variables, we already allow stuff like bitset optimizations in keeping track of dataflow properties.
We also provide a handy shim that transforms Bril Rust representation defined in [bril-rs](https://github.com/sampsyo/bril/tree/main/bril-rs) to our flattened representation. 


#### Bril Fuzzer

Rather than generating code at the Bril IR level, our fuzzer works on a higher-level abstraction with if-else and loop constructs.
This lets us generate “interesting” Bril programs with reducible CFGs and configurable nesting levels.
Although the reducibility of CFGs is not a requirement for dataflow analysis, it can only help in making our generated CFGs resemble those emitted from real programs.

For the same reason, we also limit the maximum nesting depth of basic blocks, _i.e._, real programs (beside maybe those produced at game jams) do not have treacherous levels of cyclomatic complexity.
By enforcing this, we consequently limit the number of back edges within each strongly-connected component (SCC) in the condensed CFG and the average component size.
We will discuss the relevance of SCCs later.

Our Bril fuzzer emits the text-based Bril representation via a custom pretty printer agnostic over its `std::fmt::Write` format stream.
We recommend using [bril2json-rs](https://github.com/sampsyo/bril/tree/main/bril2json-rs) for serializing large fuzzed Bril programs into JSON for better performance.

> Ethan: If I may plug my own work here, I've made my own [`bril2json`](https://crates.io/crates/bril2json) with an emphasis on good error messages.

## Parallel Dataflow Solver

There are two main phases for our parallel solver:

##### 1. Compute `KILL` and `GEN` sets in parallel

Our Bril representation represents variables with numbers, which, as previously stated, makes it easy for us to apply bitset optimization.
For block `b`, bit `i` in `in[b]` means either "definition at `function.instruction[i]` reaches `b`" (in reaching-definitions) or "variable `i` is live at `b`” (liveness).
We used a [SIMD accelerated bitset](https://docs.rs/fixedbitset/latest/fixedbitset/) implementation for efficiency. 

In both the sequential baseline and the parallel version, we only compute `KILL` and `GEN` sets once for each block before running the dataflow solver and refer them afterwards in all transfer passes.
This strategy avoids traversing the block instruction on every transfer calculation.
Both the reaching-definitions and liveness analyses share the same transfer function given `KILL` and `GEN` sets:
```
transfer(b) = (in[b] \ KILL[b]) U GEN[b]
```
Another important observation is that most of the computations for `KILL` and `GEN` are embarrassingly parallelizable:
the results of them for a particular block b does not depend on other blocks. 
In accordance, we compute them in parallel.

**Reaching-definitions**:
* `DEFS[y]`: a set of definitions of variable `y` in the entire CFG.
* `GEN[b]`: a set of local variables defined in block `b`.
* `KILL[b]`: a set of definitions that local variables defined in block `b` can kill. 
   For each definition in b, `d: y = ...`, where `d` denotes the unique instruction label (can be the offset into instructions buffer), the kill set for `d` is defined as `DEFS[y] - {d}`.

`GEN[b]` only depends on block local information, whereas `KILL[b]` requires `DEFS[y]` that depends on information from every block.
However, `DEFS[y]` can be computed with a simple map-reduce or fold-reduce in parallel: compute `DEFS[y]` for each block in parallel and merge them together by taking the union.

**Liveness**:

* `GEN[b]`: The set of variables that are used in b before any assignment in the same block.
* `KILL[b]`: The set of variables that are assigned a value in `b`.

Both `GEN[b]` and `KILL[b]` only depend on block local information. 


We used [`rayon`'s iterator adaptors](https://docs.rs/rayon/latest/rayon/) to implement this parallelization.

##### 2. Condensed CFG traversal in parallel:

We applied [Tarjan's algorithm](https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm) to decompose a CFG into a DAG of SCCs (condensed CFG) in linear time.
DAGs, of course, naturally lend themselves to parallelism.
Thus, we would apply the sequential implementation to each SCC and schedule dependent jobs for each SCC in a thread pool following the DAG dependencies.
Our current policy is simple: once the dependencies for a SCC have all been computed, we immediately submit a new worker for that component to the thread pool.
A `rayon` thread pool `Scope` is passed around between workers to allow them to recursively submit new works.
Inside each SCC, the sequential solver only follows edges between blocks in the same component, ignoring other inter-component edges. 

In the forward pass, an SCC’s input state is computed by merging the out state of its predecessor blocks in already-processed predecessor SCCs.
In the backward pass, we treat any blocks that have inter-component edges pointing to it as potential entry points.
We use the same merge methods to compute their initial states as the forward pass.

## Evaluations

We ensured the correctness of our parallel solver by comparing its results with that of the sequential solver on Bril's [core benchmarks](https://github.com/sampsyo/bril/tree/main/benchmarks/core) and fuzzed programs.

We compared the average performance between sequential and parallel solver on 20 large scaled fuzzed Bril programs, which are generated with:

```shell
bril-fuzzer --num-blocks 1024 --block-size-mean 128 --max-nesting 3
```
Bitset optimization is applied to both sequential and the parallel solver.
Therefore, the sequential baseline is somewhat parallelized with SIMD accelerated bitset implementation.
The parallel condensed CFG traversal approach is only applied to the parallel solver.
The numbers below are the total time elapsed to complete the analysis for all 20 fuzzed Bril benchmarks.
The fastest, slowest and mean metrics were collected from 10 different runs.

The experiments were conducted on an old Macbook Pro with MacOS 12.5.1 and 8 (with hyper-threading) 2GHz i5 intel CPU cores.
We fixed the number of workers of parallel solver to 4 throughout the evaluations.
We used rustc 1.87.0-nightly.

**Liveness**: 1.85x faster 
| Method     | Fastest (ms) | Slowest (ms) | Mean (ms) |
|------------|--------------|---------------|-----------|
| Parallel   | 231.6        | 233.9         | 232.7     |
| Sequential | 427.0        | 434.2         | 430.6     |


**Reaching-Definitions**: 8% slow down
| Method     | Fastest (s) | Slowest (s) | Mean (s) |
|------------|--------------|---------------|-----------|
| Parallel   | 17.40        | 24.11         | 20.76     |
| Sequential | 18.76        | 19.41         | 19.08     |

Reaching-definitions analysis reported here directly tracks the definition in a granularity of its offset into the function's instructions buffer.
We had also tried a coarser-grained version by only tracking the index of the block associated with definitions, which could give a 20x speed up compared with the fine-grained version. However, that did not change the relative performance between the sequential and parallel solver in any nontrivial way. 

**Profiling Results**:
We profiled our runs on the fuzzed programs with [samply](https://github.com/mstange/samply); surprisingly, we found that the embarrassingly parallelizable computation of `KILL` and `GEN` set actually dominates the total runtime.
The parallel dataflow phase only accounts for 30% of the runtime in liveness analysis --- and a mere 0.1% for reaching-definitions analysis.

**Remarks**:
For reaching-definitions, the profiling results indicate that parallel condensed CFG traversal approach has little impact on the final performance.
Unlike in liveness analysis, we did not see an expected speedup when parallelizing `KILL` and `GEN` computation for reaching-definitions.
The main bottleneck is the computation of `DEFS` for all variables in the CFG.
The parallel fold-reduce/map-reduce approach we applied somehow did not yield any significant speedup.

## Future work

In the parallel condensed CFG traversal phase, we currently treat all the components the same.
That is, we always submit a new intra-component sequential dataflow job to the thread pool regardless of the component's size or other potential heuristics that might influence the dataflow problem complexity.
We should have a smarter policy to decide when we should launch a dedicated thread for a new component.

We also can have a better load-balancing strategy to determine which component should run next in order to maximize the number of worker executing in parallel at every time and prevent the overall dataflow from stalling on a few unfinished SCCs.
We can use some per component heuristics, such as component size, the number of backedges within the component, out degree, etc. to precompute a better condensed CFG traversal ordering or guide the local choice at each component during traversal when picking the next to run.
We may further choose to devote more threads for large SCCs to parallelize the sequential worklist algorithm.
But we are a little bit skeptical about how far this parallel condensed CFG approach will take us given its often limited impact on analysis performance as shown in these profiling results.
