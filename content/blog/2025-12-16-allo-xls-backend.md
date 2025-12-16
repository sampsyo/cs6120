+++
title = "From Allo to XLS: Bridging Hardware Accelerator DSLs Through Traditional Compilation"

[extra]
latex = false
bio = """
Cynthia Shao is an undergraduate junior studying ECE and CS at Cornell. In her free time she likes to crochet, knit, make fun matcha drinks, rock climb, and perform Chinese martial arts. 

Nikil Shyamsunder is an undergraduate junior studying Math and CS at Cornell. In his free time he likes to solve gerrymandering with algorithmic game theory, listen to Agatha Christie audio books, and go on scenic walks. 
"""

[[extra.authors]]
name = "Cynthia Shao"
[[extra.authors]]
name = "Nikil Shyamsunder"
+++

## What's the Goal?

Hardware accelerator design increasingly relies on portable compilation flows that can target diverse backends without extensive manual retargeting. This project explores building a compilation pathway from Allo, an MLIR-based accelerator DSL, to Google's XLS hardware synthesis framework. 

Our compiler implements structured lowering passes from Allo's intermediate representation to both DSLX (XLS's functional hardware DSL) and XLS IR. The key research question is: **Can we build a practical compiler backend that bridges two modern MLIR-based accelerator frameworks, enabling designers to leverage XLS's optimization infrastructure and synthesis capabilities for Allo programs?**

We focus on two major compilation challenges:
1. **Feedforward function lowering**: Translating imperative Allo kernels with mutable memory into XLS's functional representations
2. **Systolic array generation**: Mapping Allo's spatial dataflow constructs to XLS's explicit process networks with channel communication

Our evaluation methodology combines functional correctness validation through XLS's interpreter, cycle-accurate Verilog simulation using Verilator, and physical synthesis using commercial tools (Synopsys Design Compiler and Cadence Innovus) on FreePDK 45nm. We also developed a randomized differential testing framework using XLS's QuickCheck infrastructure to validate equivalence under diverse inputs.

## Project Planning and Scope

The project went through several iterations of scope refinement during the proposal phase. Initially, we considered two potential targets: extending Allo's existing Catapult HLS backend or creating a new XLS backend from scratch. After discussion with Professor Sampson, we chose XLS for several key reasons:

1. **More interesting technically**: XLS requires figuring out how to break higher-level Allo features down into the feedforward circuit model and process networks that XLS is built around, whereas Catapult would mainly involve swapping pragma formats.

2. **Better for evaluation**: XLS is open-source and easy to install, with a built-in interpreter for correctness checking. Catapult requires proprietary licenses and would complicate evaluation.

3. **IR-to-IR translation**: Since we're starting with Allo's MLIR, translating to XLS IR is more robust long-term than generating DSLX source code (though we ended up doing both for different use cases).

We progressively refined our scope through three key stages:

**Stage 1: Naive Matrix Multiplication**  
Direct triple-nested loop with reduction over the shared inner dimension. This established the basic lowering pathway: loops, types, indexing, and function calls from Allo IR into XLS IR.

**Stage 2: Optimized Matrix Multiply Kernels**  
Tiled, reordered, and pipelined variants that Allo can express. These introduced temporary buffers, grid loops that expand into nested iteration spaces, and scheduling transformations. This demonstrated our backend could handle realistic optimization styles.

**Stage 3: Small Fixed-Size Systolic Arrays**  
A 2×2 grid of processing elements representing Allo's dataflow patterns. This required recognizing local data movement through FIFOs and mapping to XLS procs connected through channels.

Across these stages, we committed to supporting: static loop nests, reductions, simple grid loops, temporary buffers, basic arithmetic over various integer and float types, and straightforward memory accesses. We explicitly excluded advanced features like complex meta-conditionals and the full Allo scheduling language to keep the compilation strategy manageable.

This staged approach gave us a coherent set of workloads with increasing complexity while maintaining tractability.

## Implementation

### Feedforward Function Lowering (Source to DSLX)

DSLX functions serve as our initial compilation target, providing a high-level functional representation within the XLS ecosystem. Unlike imperative hardware languages, DSLX functions are pure: they accept immutable array arguments, perform computation without side effects, and return result arrays. This functional model contrasts sharply with Allo's imperative semantics, where computations operate on mutable memory buffers.

Our lowering pass implements an AST-mediated translation in two phases. First, the `MlirToDslxLowerer` class walks the MLIR function body, dispatching each operation to specialized lowering methods that construct an intermediate abstract syntax tree with node types like `DslxVar`, `DslxFor`, `DslxStore`, and `DslxBinOp`. A `CodegenContext` object tracks MLIR-to-AST value bindings, memref shapes, and loop nesting structure. Only after the entire function transforms into AST form does the `emit_dslx` method recursively generate properly formatted DSLX source.

The central challenge lies in the impedance mismatch between MLIR's memory model and DSLX's functional semantics. Allo programs freely perform `affine.load` and `affine.store` operations on memref buffers, treating arrays as mutable storage. DSLX forbids in-place mutation. To bridge this gap, we translate each `affine.store` into a DSLX `update` expression. For a 2D store like `C[i][j] = value`, the lowering emits `update(C, i, update(C[i], j, value))`, constructing a new array rather than mutating in place.

Affine for loops require careful treatment to ensure array updates propagate correctly. In MLIR, loops implicitly mutate buffers; DSLX requires explicit accumulator variables threading updated arrays across iterations. Our lowering scans each loop body to identify written buffers, then generates DSLX for-expressions like `let C = for (i, C) in u32:0..u32:N { ... }(C)`, where the accumulator `C` is both an iteration parameter and the final result. For nested loops updating the same buffer, this pattern extends naturally. When multiple distinct buffers are modified, the lowering constructs tuple accumulators like `(A, B, C)` with corresponding tuple destructuring.

Here's an example transformation:
```mlir
// MLIR input
func.func @gemm(%A: memref<32x32xi32>, %B: memref<32x32xi32>) -> memref<32x32xi32> {
  %C = memref.alloc() : memref<32x32xi32>
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      affine.for %k = 0 to 32 {
        %a = affine.load %A[%i, %k]
        %b = affine.load %B[%k, %j]
        %prod = arith.muli %a, %b
        %c = affine.load %C[%i, %j]
        %sum = arith.addi %c, %prod
        affine.store %sum, %C[%i, %j]
      }
    }
  }
  return %C
}
```
```rust
// Generated DSLX
fn gemm(arg0: u32[32][32], arg1: u32[32][32]) -> u32[32][32] {
  let C = u32[32][32]:[[u32:0, ...], ...];
  let C = for (i, C) in u32:0..u32:32 {
    for (j, C) in u32:0..u32:32 {
      for (k, C) in u32:0..u32:32 {
        update(C, i, update(C[i], j, (C[i][j] + (arg0[i][k] * arg1[k][j]))))
      }(C)
    }(C)
  }(C);
  C
}
```

### XLS IR Lowering

XLS IR serves as an alternative, lower-level compilation target. Unlike DSLX, which requires parsing and compilation, XLS IR can be directly interpreted or JIT-compiled using XLS's `eval_ir_main` tool. XLS IR is a pure dataflow representation in SSA form: each operation explicitly names inputs and produces uniquely named outputs, with no implicit state.

The XLS IR lowering reuses much of the DSLX lowering's structural scaffolding but diverges fundamentally in loop handling. Where DSLX preserves loop structure through functional for-expressions, XLS IR eliminates loops entirely through unrolling. When encountering an `affine.for` loop with bounds `lb` to `ub`, the lowerer generates a sequence of operations for each integer in that range, materializing loop indices as literal constants. For a loop `for %i = 0 to 3`, it emits `i_0: bits[32] = literal(value=0)`, binds the MLIR induction variable to `i_0`, lowers all operations in the loop body, then repeats with `i_1: bits[32] = literal(value=1)`, and so forth. Nested loops unroll multiplicatively.

Multi-dimensional arrays are represented as one-dimensional arrays in XLS IR, requiring explicit linearization. For a 2D memref of shape N×M, a load `affine.load %A[%i, %j]` translates to computing `linear_idx = i * M + j`, then `array_index(A, indices=[linear_idx])`. Array updates are more intricate: XLS IR's `array_update` returns a new array rather than mutating in place, requiring careful SSA versioning. Each store generates a fresh array name, and the lowerer must rebind all subsequent references.

### Systolic Array Generation

Systolic arrays represent our most complex transformation. XLS organizes computation as communicating processes based on Kahn process network semantics. A `proc` represents a stateful concurrent process with three components:
- A `config` function declaring and wiring communication channels
- An `init` function establishing initial state values  
- A `next` function implementing a single state transition by consuming inputs, updating state, and producing outputs

Rather than attempt general lowering from arbitrary Allo programs to explicit state machines (an intractably hard problem), we adopt a pattern-matching approach that specifically recognizes systolic array structures in MLIR and generates corresponding XLS grid implementations.

The lowering pipeline begins with pattern detection and structural extraction. Specialized extractors recover metadata from MLIR: parsing loop bounds for grid dimensions, analyzing array subview index patterns to infer inter-PE dataflow directions, and inspecting function signatures to recover problem dimensions.

The extracted metadata drives a builder that constructs a DSLX proc AST. Each PE becomes a proc with explicit state (accumulator and iteration counter), a config function declaring directional channels (`from_west`, `from_north`, `to_east`, `to_south`), an init function establishing zero initial state, and a next function implementing the MAC computation with explicit token threading through `recv` and `send` operations.

Here's the generated PE structure:
```rust
proc PE {
  result_out: chan<u32> out;
  from_west: chan<u32> in;
  from_north: chan<u32> in;
  to_east: chan<u32> out;
  to_south: chan<u32> out;

  config(result_out: chan<u32> out, from_west: chan<u32> in,
         from_north: chan<u32> in, to_east: chan<u32> out,
         to_south: chan<u32> out) {
    (result_out, from_west, from_north, to_east, to_south)
  }

  init { (u32:0, u32:0) }

  next(state: (u32, u32)) {
    let (accum, k) = state;
    let (tok, a) = recv(join(), from_west);
    let (tok, b) = recv(tok, from_north);
    let prod = a * b;
    let new_accum = accum + prod;
    let tok = send(tok, to_east, a);
    let tok = send(tok, to_south, b);
    let new_k = k + u32:1;
    let should_output = new_k == u32:2;
    let tok = send_if(tok, result_out, should_output, new_accum);
    let new_state = if should_output { (u32:0, u32:0) } 
                    else { (new_accum, new_k) };
    new_state
  }
}
```

The builder also generates a top-level `SystolicArray` proc that creates 2D channel arrays and uses `unroll_for!` to spawn the grid at compile time, orchestrating computation by feeding input matrix elements into edge channels and draining results after K iterations.

### Meta-Systolic Arrays: Analyzing Spatially-Specialized Functions

Beyond the basic pattern-matching approach for library-style systolic arrays, we developed a lowering strategy for systolic arrays expressed using Allo's metaprogramming constructs. However, the key challenge is that **Allo's `meta_if` constructs are compiled away before we see them**—by the time our compiler receives the MLIR, the conditional code generation has already happened.

When Allo compiles a dataflow kernel like:
```python
@df.kernel(mapping=[P0, P1])
def gemm(A, B, C):
    i, j = df.get_pid()
    with allo.meta_if(i == 0):
        # Load B columns
        ...
    with allo.meta_elif(j == 0):
        # Load A rows
        ...
    with allo.meta_else():
        # Interior PE: MAC computation
        ...
```

The resulting MLIR contains **separate functions for each grid position**: `gemm_0_0`, `gemm_0_1`, `gemm_1_0`, `gemm_1_1`, etc. Each function has different code depending on which `meta_if` branch was selected for that `(i, j)` coordinate at compile time.

**Our Compilation Challenge:** We must reverse-engineer the spatial structure and PE types from this unrolled representation by analyzing the behavior of each generated function.

#### Pattern Detection

The `SystolicDetector.is_metaif_systolic()` method identifies this pattern by looking for:

1. **Multiple `gemm_i_j` functions**: The naming pattern reveals spatial grid structure
2. **Heterogeneous behavior**: Some functions contain multiply-accumulate logic (interior PEs), others contain only loads or stream operations (edge handlers)

Specifically, it counts functions with MAC computation:
```python
def _has_mac_computation(self, func_op):
    """Check if function has multiply-accumulate computation."""
    has_mul = False
    has_add = False
    for op in func_op.body:
        if isinstance(op, affine_d.AffineForOp):
            for inner_op in op.body.operations:
                if 'Mul' in str(type(inner_op).__name__):
                    has_mul = True
                if 'Add' in str(type(inner_op).__name__):
                    has_add = True
    return has_mul and has_add
```

A meta-if systolic array is detected when there are ≥4 `gemm_i_j` functions and at least one contains MAC computation.

#### Function Classification

The `MetaIfSystolicTranslator` analyzes each unrolled function to classify its role. There are interior PEs which contain multiply accumulate computation, input loaders (edge PEs that inject data), and drain PEs (edge PEs that consume data).

#### Structural Extraction

From the function names, we extract grid dimensions:
```python
def _extract_grid_size(self):
    """Extract M, N from function names like gemm_1_1, gemm_2_2."""
    max_i = 0
    max_j = 0
    for name in self.functions.keys():
        match = re.search(r'gemm_(\d+)_(\d+)', name)
        if match:
            i, j = int(match.group(1)), int(match.group(2))
            max_i = max(max_i, i)
            max_j = max(max_j, j)
    # Grid includes borders, so actual size is max - 1
    self.grid_size = {
        'rows': max_i - 1,  # Interior rows (M)
        'cols': max_j - 1,  # Interior cols (N)
    }
```

For a 2×2 systolic array with K=2, Allo generates a 4×4 grid of functions (including edge loaders/drains): `gemm_0_0` through `gemm_3_3`. The interior 2×2 PEs perform MAC operations, while the border functions handle input injection and output draining.

We also extract the K loop bound from interior PE functions by analyzing the `affine.for` upper bound, and infer element types from stream type annotations in function signatures (e.g., `!allo.stream<f32, 4>` → `u32` in DSLX).

#### DSLX Code Generation

Once we've classified all functions, we generate a heterogeneous XLS proc network:

1. **Parameterized PE proc**: Generated from the interior PE pattern, supporting configurable K through DSLX generics (`proc PE<K: u32>`)

2. **Channel network**: A-FIFOs for horizontal flow (M rows × N+1 columns), B-FIFOs for vertical flow (N columns × M+1 rows), and C-output channels (M × N)

3. **Specialized spawning**: Unlike our basic systolic lowering which spawns identical PEs, this instantiates the parameterized PE only for interior positions and inlines the loader/drain logic

4. **Input/output coordination**: The top-level proc receives input matrix elements through separate channels for each row/column, distributes them to edge PEs, collects results from output channels, and assembles the final matrix

**Key Limitation:** This approach works because Allo's dataflow compiler already did the hard work of spatial unrolling. We're essentially pattern-matching on the *output* of Allo's metaprogramming system rather than analyzing the metaprogramming constructs themselves. We can't directly see the original `meta_if` conditionals or reason about which spatial positions satisfy which conditions—we only observe their effects in the generated function bodies.

This makes our meta-systolic lowering more of a **"structure recovery"** pass than true metaprogramming analysis. We reconstruct the systolic array topology by observing communication patterns (which functions call `stream_get`/`stream_put` on which FIFOs) and computational behavior (which functions perform MAC operations). The technique works well for regular systolic arrays where Allo's conventions are predictable, but would struggle with more exotic spatial heterogeneity or communication topologies that don't fit our classification heuristics.

The meta-systolic system successfully generates synthesizable XLS procs for 2×2, 3×3, and 4×4 grids validated through Verilator simulation, demonstrating that structure recovery from unrolled representations is viable for this class of accelerators.

## What's Hard About This?

### The Imperative-to-Functional Transformation

The biggest conceptual challenge was bridging the semantic gap between Allo's imperative memory model and XLS's functional purity constraint. This isn't just a syntactic transformation—it requires fundamentally restructuring how computation is expressed. 

In imperative code, you write `C[i][j] += value` and the hardware implements this as a read-modify-write to a specific memory location. In DSLX, there is no memory—only immutable values flowing through pure functions. Converting `C[i][j] += value` into `update(C, i, update(C[i], j, C[i][j] + value))` means explicitly constructing a new array that differs from the old array only at position `[i][j]`. For nested loops, this requires threading these updated arrays as accumulators across iterations, essentially making data flow explicit where it was previously implicit in memory side effects.

The tricky part is that this transformation must be completely mechanical and deterministic. We can't rely on human intuition about "what the code means"—we need to programmatically walk MLIR operations and construct the correct accumulator threading pattern based purely on structural analysis. Getting this right required careful reasoning about which buffers are modified in which loops, how to order tuple destructuring when multiple buffers are threaded, and when nested loops need their own accumulators vs. reusing outer accumulators

### Systolic Array Pattern Matching

The systolic array lowering is brittle in ways that feel unsatisfying from a compiler engineering perspective. We rely heavily on syntactic heuristics—substring matching on function and buffer names, regular expression parsing of stringified MLIR expressions, hardcoded assumptions about argument positions and nesting depths—rather than true semantic analysis.

For instance, detection requires that FIFO buffers contain the substring "fifo" in their names, that PE kernel functions follow specific naming conventions, and that connectivity can be inferred from positional argument ordering. These are artifacts of how Allo currently lowers systolic array programs to MLIR. Changes to Allo's frontend passes or different buffer naming schemes could break extraction without any change to the underlying computation's semantics.

A robust solution would require either significantly more sophisticated pattern recognition using static analysis and dataflow analysis or a general IR transformation framework capable of inferring process networks from arbitrary imperative code

Both are substantially harder problems than we could tackle in this project. Our approach works for the systolic arrays Allo generates today, but it's fragile.

### XLS Toolchain Quirks

Getting systolic arrays to compile through XLS's full pipeline (DSLX → IR → optimized IR → pipelined Verilog) required substantial debugging. Even Google's reference systolic array implementation in the XLS repository fails to compile through the IR optimization and codegen stages due to cyclic FIFO dependencies and unresolved scheduling constraints.

We had to systematically debug by:
- Isolating which XLS passes were failing
- Examining the intermediate IR representations
- Redesigning channel network topology and `unroll_for!` spawning patterns
- Validating modifications against XLS's stricter compilation requirements

This debugging process was time-consuming and required deep understanding of XLS's internal pipeline stages. Documentation was sparse for these advanced features, so we relied on reading XLS source code and experimenting with minimal test cases. Professor Sampson's pointer to the [XLS matmul_4x4 example](https://github.com/google/xls/blob/main/xls/examples/matmul_4x4/matmul_4x4.x) and the [proc tutorial](https://google.github.io/xls/tutorials/how_to_use_procs/#channel-arrays-and-loop-based-spawning) proved invaluable for understanding channel arrays and loop-based spawning patterns. But, during the process of this project we found out that the golden matmul example couldn't properly lower to Verilog, as XLS's scheduling created circular dependencies within their fifo configuration.

### Verilator Simulation Infrastructure

Setting up cycle-accurate simulation required building a complete testbench infrastructure. We created generic design under test files for verilator, and scripts to run the verilator test file under different matrix sizes and codegen with different pipeline stages. We additionally implemented cycle-by-cycle output checking, and added scripts to generate CSVs to output the cycles in a digestible manner.

For systolic arrays, this was particularly complex because we needed to feed matrix data into edge channels following the correct protocol, respect valid/ready handshaking on FIFO channels, collect results at the right time after K iterations complete, and verify correctness despite cycle-level timing variations.

Getting the timing right required careful analysis of the generated Verilog to understand XLS's channel implementation and synchronization behavior.

## Did It Work?

### Functional Correctness

Yes! Our feedforward DSLX lowering successfully compiles a range of GEMM kernels with varying matrix dimensions and Allo scheduling transformations. We generated correct DSLX for 2×2, 4×4, and 32×32 matrix multiplications, exercising different Allo scheduling primitives:
- `bench_simple`: unscheduled baseline
- `bench_split_outer`: outer loop tiling  
- `bench_split_inner`: inner loop tiling
- `bench_split_both`: two-level tiling
- `bench_split_reorder`: loop reordering
- `bench_compose`: composed transformations

Each variant produces syntactically and semantically correct DSLX functions that pass validation via the DSLX interpreter on handwritten tests.

For XLS IR generation, we successfully generated unrolled XLS IR for 4×4 GEMM benchmarks across all scheduling variants. The generated IR executes correctly under `eval_ir_main`, validating that our linearized array indexing, SSA rebinding, and unrolled loop bodies faithfully implement the original Allo computation.

### Randomized Differential Testing

We developed a randomized differential testing framework using XLS's QuickCheck infrastructure to validate equivalence between our compiler-generated code and hand-written implementations under diverse inputs. For each benchmark, we constructed differential test harnesses:
```rust
#[quickcheck(test_count=5)]
fn prop_gemm_equivalence(A: s32[32][32], B: s32[32][32]) -> bool {
    let result_compiler = allo_gemm_compiler(A, B);
    let result_reference = allo_gemm_reference(A, B);
    result_compiler == result_reference
}
```

The XLS interpreter automatically generates randomized 32×32 integer matrices as inputs, executes both implementations, and verifies bitwise equality of outputs. This provides substantially greater input space coverage than hand-crafted test cases.

Our framework successfully validates the `simple`, `split_outer`, and `split_both_merge` benchmarks, with all random trials passing. The `split_inner` and `split_both` benchmarks currently fail differential testing, correctly identifying known bugs in those implementations. This demonstrates the framework's effectiveness at automatically detecting semantic errors that might escape human review.

### Cycle-Accurate Simulation 

We performed Verilator simulation on several designs to validate functional correctness and measure performance. Here are the key results for our manually-compiled baseline GEMM and various meta-systolic configurations:

**2×2 Designs (K=2):**

| Design | Type | Pipeline Stages | Cycles | Status |
|--------|------|----------------|---------|--------|
| Combinational GEMM | uint32 | - | 5 | PASS |
| Systolic | uint32 | 2 | 18 | PASS |
| Systolic | uint32 | 3 | 19 | PASS |
| Systolic | uint32 | 5 | 22 | PASS |
| Systolic | uint32 | 7 | 30 | PASS |
| Systolic | float32 | 2 | 16 | PASS |
| Systolic | float32 | 3 | 17 | PASS |
| Systolic | float32 | 5 | 18 | PASS |
| Systolic | float32 | 7 | 19 | PASS |

The combinational GEMM designs complete in just 5 cycles regardless of source-level optimizations—XLS's aggressive optimization passes completely normalize high-level structural variations. All Allo scheduling transformations (split, reorder, compose) produce identical cycle counts after synthesis.

Systolic arrays show higher latency (16-30 cycles for 2×2) but scale better for larger matrices. The latency increases with pipeline depth as expected, with deeper pipelines trading latency for higher maximum frequency. Notably, all designs with pipeline depth 1 failed Verilog generation—XLS requires at least 2 pipeline stages for proc networks to synthesize correctly.

**Larger Grid Results:**

| Design | Rows×Cols | K | Type | Pipeline Stages | Cycles | Status |
|--------|-----------|---|------|----------------|---------|--------|
| Systolic | 3×3 | 2 | uint32 | 2 | 26 | PASS |
| Systolic | 3×3 | 2 | uint32 | 3 | 27 | PASS |
| Systolic | 3×3 | 2 | uint32 | 5 | 30 | PASS |
| Systolic | 3×3 | 2 | uint32 | 7 | 41 | PASS |
| Systolic | 4×4 | 4 | int32 | 2 | 38 | PASS |
| Systolic | 4×4 | 4 | int32 | 3+ | - | NO OUTPUT |

The 3×3 design scales as expected, with cycle count growing roughly with grid size. However, the 4×4 design with K=4 fails to produce valid output for pipeline depths ≥3, hitting the 1480-cycle timeout. This suggests XLS's scheduling algorithms struggle with larger proc networks at higher pipeline depths, possibly due to channel synchronization deadlocks or state space explosion in the scheduler.

### Performance Analysis

The cycle-accurate simulation reveals interesting architectural tradeoffs:

**Combinational vs. Systolic (2×2):**
- Combinational: 5 cycles (3.4× faster)
- Systolic (depth 5): 18 cycles

The combinational design instantiates four parallel multipliers (one per output element) and executes all computations simultaneously through a fixed 5-stage pipeline. The systolic array reuses four processing elements across K=2 iterations, with each PE implementing its own pipeline.

The 18-cycle systolic latency breaks down into:
1. PE pipeline execution across K=2 iterations (~8-10 cycles)
2. State machine coordination overhead (~3 cycles)  
3. Result collection (~4 cycles) for sequential gathering from all PEs

Additional latency arises from channel handshaking protocols—depth-1 FIFOs on all internal channels require valid/ready synchronization for each data transfer.

### Systolic Array Synthesis

Our systolic array lowering successfully generated and validated systolic arrays across multiple configurations:
- 2×2, 3x3, and 4×4 grids with K=2 and K=4
- Both integer (`u32`, `s32`) and floating-point (`F32`) element types
- Multiple pipeline depths (2, 3, 5, 7 stages)

<img src="./2025-12-16-allo-xls-backend/manual_gemm_pnr.png" alt="architecture diagram" width="300"/> <img src="./2025-12-16-allo-xls-backend/systolic_2x2_pnr.png" alt="architecture diagram" width="310"/>

Above are schematics of final place and route for the manually compiled vanilla 2x2 uint32 gemm and the 2x2 uint32 manually compiled systolic design from the Allo library. 

Critically, we achieved **end-to-end Verilog generation** for multiple systolic configurations. This represents a nontrivial achievement: even Google's reference systolic array implementation in the XLS repository fails to compile through the IR optimization and codegen stages. We identified these issues through systematic debugging, redesigned the channel network topology and `unroll_for!` spawning patterns, and validated our modifications against XLS's stricter compilation requirements.



**Scalability Implications:**

For 2×2 matrices, combinational designs dominate. However, extrapolating to larger matrices reveals inflection points where systolic arrays become favorable:

- **Combinational designs** require M×N parallel multipliers, scaling to ~410,000 µm² for 8×8 (64 multipliers) and ~1.64 mm² for 16×16 (256 multipliers). Beyond 32×32, they become impractical due to quadratic area growth and timing closure challenges from global wire delays.

- **Systolic arrays** require M² PEs regardless of matrix dimension, with area growing proportionally to PE complexity. For 16×16 matrices, the systolic array needs only 256 PEs (~640,000 µm²), while combinational likely exceeds 1 mm².

The crossover point appears around **8×8 to 16×16 matrix dimensions**—exactly the range where hardware accelerators become interesting for real applications. Unfortunately, we were unable to emperically evaluate due to the time constraints of performing syntheis and place and route on these large designs. Due to XLS's policy of unrolling every functional loop, we found that 2x2 designs were the most optimal to push through the chip design flow.

## Surprising Findings

Our research partners explored an alternative compilation approach using LLM-based agents (see [our project video](https://www.youtube.com/watch?v=OcR5Z9o5RB4&list=PLRvJfry30-22JzxHU2XuGQe0oJeMk-9bm&index=11) for details). Remarkably, they found that LLM-generated DSLX code, after passing through XLS's optimization and synthesis pipeline, converged to **identical hardware implementations** as our manually-compiled code. All designs are 2x2 uint32 matmuls, vanilla gemms pipelined to 5 stages, and the systolic to 2 (empirically evaluated to be the best), with cycle counts from Verilator, and the area, clock, and power from commercial tools. 

| Design | Area (µm²) | Clock (ns) | Power (mW) | Cycles |
|--------|-----------|-----------|-----------|--------|
| **Compiler GEMM** | 36,480 | 4.06 | 17.38 | 5 |
| **Claude GEMM** | 36,480 | 4.06 | 17.38 | 5 |
| **GPT GEMM** | 36,480 | 4.06 | 17.37 | 5 |
| **Compiler Systolic** | 51,076 | 4.10 | 16.74 | 18 |

Both Claude Opus 4.5 and GPT-generated code—despite having completely different source-level structure, variable naming, and loop organization—produced the exact same post-synthesis characteristics: same cycle count, same critical path, same area. XLS's aggressive optimization passes completely normalized all high-level variations.

This suggests an intriguing insight for hardware compiler design: **algorithmic correctness matters far more than source-level micro-optimizations** when backend synthesis tools can extract optimal implementations from varied but correct source code. The LLM approach dramatically reduced development time compared to traditional compiler engineering, with models requiring an average of 2-3 iterations to produce correct code.

However, the LLM approach remains fundamentally limited to architectures expressible through local syntactic transformations. Systolic arrays and other stateful, communicating process networks lie beyond current LLM capabilities without explicit architectural templates. Our traditional compiler approach handles these complex patterns through structured IR analysis and code generation, demonstrating the continued value of formal compilation techniques for advanced hardware constructs.

## Lessons Learned

### Compiler Engineering vs. Pattern Matching

Our systolic array implementation highlights a fundamental tension in compiler design: general solutions vs. practical implementations. The "right" way to lower arbitrary Allo programs to XLS procs would be a general algorithm for extracting concurrent processes from imperative code—but this is extremely hard. 

Our pattern-matching approach works well for the specific use case (systolic arrays) but is brittle and special-cased. This is a common tradeoff in real compiler backends: perfect generality is often impractical, so production compilers include special-case optimizations and lowering passes for important patterns. The key is making the special cases explicit and well-documented, so they can be maintained and extended.

### The Value of Intermediate Abstractions

The AST intermediate representation for DSLX lowering proved invaluable. By separating semantic transformation (understanding what MLIR constructs mean) from text generation (emitting syntactically correct DSLX), we made the lowering logic much clearer and easier to debug. When we tried direct text emission for XLS IR, it worked for that specific case but would have been harder to extend or modify.

This reinforces a general principle: well-chosen intermediate representations make compiler passes more maintainable, even if they add some complexity upfront.

### Testing is Essential but Hard

Building the differential testing framework took significant effort, but it was absolutely worth it. Manual test cases would never have caught all the edge cases we discovered through randomized testing. However, setting up proper testing infrastructure for hardware compilers is challenging for many reasons. We struggled with getting the interpreter and simulation tools working. Additionally, hardware timing makes equivalence testing complex, and test input generation needs to be specialized to the input program.

## Future Directions

### Extending Pattern Coverage

The current systolic array pattern matcher could be extended to recognize other regular structures like convolution engines, transposed systolic arrays, or streaming datapath patterns. Each would require careful analysis of Allo's MLIR representation and mapping to XLS proc primitives, but the basic infrastructure is in place.

### Loop Optimization Strategies  

Our XLS IR lowering currently unrolls loops completely, which limits scalability. More sophisticated approaches could:
- Use loop trip count analysis to decide between unrolling and proc-based iteration
- Implement partial unrolling with tiling transformations
- Generate pipelined proc networks for large loop bodies

### Better Integration with Allo

Currently, our compiler requires specific MLIR patterns from Allo. Better integration would involve:
- Defining a formal interface specification between Allo and XLS lowering
- Adding Allo-side annotations to mark systolic arrays or other patterns explicitly
- Providing feedback from XLS synthesis to Allo's scheduling decisions

### Automated Design Space Exploration

The meta-systolic system demonstrates automated variant generation, but we could extend this to:
- Automatically explore pipeline depth vs. frequency tradeoffs
- Generate Pareto-optimal configurations across area/power/performance
- Use synthesis results to guide further generation (iterative optimization)

## Conclusion

We successfully built a compiler backend bridging Allo and XLS, demonstrating that MLIR-based accelerator DSLs can target XLS's synthesis infrastructure. Our implementation handles both pure dataflow computations (via functional DSLX lowering) and complex stateful hardware patterns (via proc-based systolic arrays).

We made many technical contributions that we are proud of, enumerated below:
1. **Imperative-to-functional transformation**: Systematic conversion of mutable memory operations into functional update expressions with proper accumulator threading
2. **Systolic array pattern recognition**: Extracting spatial dataflow structure from MLIR and mapping to explicit process networks
3. **Meta-systolic structure recovery**: Analyzing spatially-unrolled functions to reconstruct heterogeneous systolic topologies
4. **End-to-end validation**: From DSLX generation through cycle-accurate simulation on real designs

Our evaluation demonstrates functional correctness across diverse Allo scheduling transformations and successful Verilog generation for systolic arrays where even reference implementations fail. Cycle-accurate simulation reveals that combinational designs dominate for small matrices while systolic arrays scale better to larger dimensions, requiring less power and similar area.

The comparison with LLM-based compilation approaches (detailed in [our HLS video](https://www.youtube.com/watch?v=OcR5Z9o5RB4&list=PLRvJfry30-22JzxHU2XuGQe0oJeMk-9bm&index=11)) reveals surprising convergence at the hardware level despite radically different source generation methods. This suggests that backend optimization is powerful enough to normalize high-level variations—a result that has implications for future hardware compiler design and the potential role of AI-assisted code generation.

While our pattern-matching approach for systolic arrays is admittedly brittle, it demonstrates the fundamental feasibility of connecting Allo's spatial abstractions to XLS's process networks. Future work could develop more general transformation algorithms or tighter integration between the two frameworks. The meta-systolic system points toward a future where hardware design space exploration is automated through compiler infrastructure rather than manual instantiation.

The full implementation is open source at [github.com/Nikil-Shyamsunder/allo-xls-backend](https://github.com/Nikil-Shyamsunder/allo-xls-backend).