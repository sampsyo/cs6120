+++
title = "Final Project: Parallel Dataflow Analysis"
[extra]
bio = """
  Ethan Uppal Cornell CS '27
  Zihan Li Cornell CS '25
"""
[[extra.authors]]
name = "Ethan Uppal"

[[extra.authors]]
name = "Zihan Li"
+++
## Background
Dataflow analysis with the worklist algorithm can be a bottleneck for compilation speed, especially for JITs:
```rust
In[entry] = init
out[*] = init
Worklist = all blocks
While worklist is not empty:
	B = pick any block from worklist
	In[b] = merge(out[p] for every predecessors p of b)
	Out[b] = transfer(b, in[b])
	If out[b] changed:
		Worklist += successors of b
```
In this [project](https://github.com/zihan0822/para-dflow), we built a parallel dataflow solver in Rust with bitset optimizations for our flattened Bril IR. We parallelized the KILL and GEN set computation and the condensed cfg traversal process. We focused on one forward pass analysis: reaching definition and one backward pass analysis: liveness analysis in particular. 


## Preparations
#### Flattened Bril Representation
We implemented a flattened representation for Bril to get rid of fragmented heap references in previous Bril representations implemented in [bril-rs](https://github.com/sampsyo/bril/tree/main/bril-rs). Here are some of our flattened equivalents. 
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

With this flattened representation, we hope to isolate the performance increase to just the dataflow analyses. It also simplifies things by tying all references’ lifetime to the program. We also provide a handy shim that transforms bril’s official repr to our flattened repr. 


#### Bril Fuzzer
Rather than generating code at the Bril IR level, our fuzzer works on AST level with if-else and loop constructs. This lets us generate “interesting” bril programs with reducible CFGs and configurable nesting levels. Although the reducibility of cfg is not a requirement for dataflow analysis, we hope to fuzz IRs that resemble those emitted from real programs.

For the same reason, we also limit the maximum nesting depth of basic blocks. In practice, most of the real-world programs won’t have loops that go over three levels deep. By enforcing this, we also limit the number of back edges within each SCC and the average component size. 

(we recommend [bril2json-rs](https://github.com/sampsyo/bril/tree/main/bril2json-rs) for serializing large fuzzed bril programs, the default [python impl](https://github.com/sampsyo/bril/tree/main/bril-txt) for that is sometimes too slow)

Our [bril fuzzer](https://github.com/zihan0822/para-dflow/tree/main/bril-fuzzer), [flattened bril repr](https://github.com/zihan0822/para-dflow/tree/main/bril) and the [parallel solver](https://github.com/zihan0822/para-dflow/tree/main/bril-analysis) are all open sourced on [Github](https://github.com/zihan0822/para-dflow/tree/main)

## Parallel Dataflow Solver
There are two main phases for our parallel solver:
##### 1. Compute KILL and GEN set in parallel
Besides flattening, our new bril representation also assigns each variable a number (zero-indexed per function) instead of strings, which makes it easy for us to apply bitset optimization. For block b, bit `i` in `in[b]` means either “definition at `function.instruction[i]`” reaches b (reaching definition) or “variable `i` is live at b” (liveness analysis). We used a [SIMD accelerated bitset](https://docs.rs/fixedbitset/latest/fixedbitset/) implementation for efficiency. 

For both the sequential baseline and the parallel version, we only compute KILL and GEN sets once for each block before running the dataflow solver and use them afterwards in all transfer passes. This avoids re-iterating block’s instruction on every transfer whenever the `in[b]` is changed. Both the reaching definition  and liveness analysis share the same transfer function given KILL and GEN set:
```
transfer(b) = (in[b] \ KILL[b]) U GEN[b]
```
Another important observation is that: most of the computations for KILL and GEN are embarrassingly parallelizable. 


**Reaching definition**:
`GEN[b]`: a set of local variables defined in block b
`KILL[b]`: for definition `d: y = … in b, KILL[b][d] = DEFS[y] - {d}`
`GEN[b]` only depends on block local info. `KILL[b]` requires `DEFS[y]` across every block, while `DEFS[y]` can be computed with a simple map-reduce or fold-reduce (however, empirically, we found that fold-reduce/map-reduce has worse performance than the sequential baseline in our setting)


**Liveness analysis**:
`GEN[b]`: The set of variables that are used in b before any assignment in the same block.
`KILL[b]`: The set of variables that are assigned a value in b
Both `GEN[b]` and `KILL[b]` only depend on block local info. 

We parallelize KILL and GEN computation with [rayon's par_iter](https://docs.rs/rayon/latest/rayon/). 




##### 2. Condensed CFG traversal in parallel:
We applied [Tarjan's algorithm](https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm) to decompose a CFG into a DAG of SCCs (condensed CFG) in linear time. DAGs, of course, naturally lend themselves to parallelism. Thus, we would apply the sequential implementation to each SCC and schedule dependent jobs for each SCC in a thread pool following the DAG dependencies. Our current policy is simple: once the dependencies for a SCC have all been computed, we immediately submit a new worker for that component to the thread pool. A rayon thread pool Scope is passed around between workers to allow them to recursively submit new works. Inside each SCC, the sequential solver only follows edges between blocks in the same component, ignoring other inter-component edges. 

In the forward pass, an SCC’s input state is computed by merging the out state of its predecessor blocks in already-processed predecessor SCCs. In the backward pass, we treat any blocks that have inter-component edges pointing to it as potential entry points. We use the same merge methods to compute their initial states as the forward pass.


## Evaluations
To test the correctness, we compare the results of sequential and parallel solver on core benchmarks and fuzzed programs to make sure they agree. 

We compare the average performance between sequential and parallel solver (`#workers = 4`) on 20 large scaled fuzzed bril programs, which are generated with:

```shell
bril-fuzzer –-num-block 1024 –-block-size-mean 128 –-max-nesting 3
```

The sequential baseline is somewhat parallelized with SIMD accelerated bitset implementation. 

**Liveness Analysis**: 1.85x faster 
| Method     | Fastest (ms) | Slowest (ms) | Mean (ms) |
|------------|--------------|---------------|-----------|
| Parallel   | 231.6        | 233.9         | 232.7     |
| Sequential | 427.0        | 434.2         | 430.6     |


**Reaching Def**: 8% slow down
| Method     | Fastest (s) | Slowest (s) | Mean (s) |
|------------|--------------|---------------|-----------|
| Parallel   | 17.4        | 24.11         | 20.76     |
| Sequential | 18.76        | 19.41         | 19.08     |



**Profiling Results**:
We profiled our runs on the fuzzed programs with [samply](https://github.com/mstange/samply), surprisingly we found that the embarrassingly parallelizable computation of KILL and GEN set actually dominates the total runtime. The parallel solver itself only accounts for 30% of the runtime in liveness analysis, and a mere 0.1% for reaching definition. 



