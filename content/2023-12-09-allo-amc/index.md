+++
title = "Frontend Integration for AMC (Accelerator Memory Compiler)"
[extra]
bio = """
  Matthew Hofmann is a 2nd year Ph.D. student researching design automation for reconfigurable hardware.<br>
  Yixiao Du is a 3rd year Ph.D. student researching hardware acceleration for graph processing and sparse linear algebra.
"""
[[extra.authors]]
name = "Matthew Hofmann"
[[extra.authors]]
name = "Yixiao Du"
+++

## Introduction

For our final course project, we have integrated our HLS (high-level sythesis) compiler, AMC, with a Python frontend, called Allo. In the end, we are able to compile and simulate custom hardware designs with extremely concise design descriptions and short compile times. As a consequence, we greatly reduce the design effort required for new accelerators as well as offer a convenient tool for functional verification. In this report, we walk through an example design with our tool flow, offer insight into how the underlying compiler works, and finally evaluate some benchmarks with latency and resource utilization measurements.

### Hardware Accelerators

When CPU and GPU optimizations have been exhausted, custom hardware accelerators are the main option for improving the performance and energy efficiency of compute intensive applications. However, the design effort required to create a new accelerator is often too high for the average programmer. This is because the design process requires a deep understanding of the underlying hardware architecture, how it can be used, and how the CAD tools behave. For example, the designer must understand how to map their algorithm to the floorplan of the hardware and how to optimize for the functional units present in architecture (e.g., LUTs, DSPs on FPGAs). Of course, debugging is its own challenge when designing hardware. Finally, the designer also needs to battle with inadequate vendor tools, which are used to synthesize the hardware description language (HDL) and run behavioral and cycle-accurate simulations. In the end, the accelerator designer must be an expert in a very tall stack of technologies, spanning both hardware and software. As a result, bringing up a new accelerator is a slow, tedious, and typically expensive process.

### High-Level Synthesis

High-Level Synthesis (HLS) is one such solution to solve the problem of hardware design productivity. The general idea is to raise the level of abstraction from the commonly-used register transfer level (RTL) (e.g. Verilog) to a higher-level language like C/C++. This allows the designer to focus more at the algorithm level and less on the hardware details, such as interfaces and timing. The "magic" of HLS compilers is that it can infer a gate-level mapping from the high-level language to the underlying hardware. The main steps of generating this mapping are scheduling, resource allocation, and binding. In the end, the outputted RTL code from HLS can be synthesized by downstream tools. However, current HLS tools greatly fall short in their promise to free designers from thinking at the architecture level. They rely on special directives (e.g., C++ pragmas) from the designer to guide the optimization process. Oftentimes, the designer must even rewrite their code to fit the HLS compiler's model of computation.

### MLIR and Incubator Projects

Reinventing HLS with advanced compiler techniques is an active area of research. There are many outstanding HLS tools/frameworks such as [TAPA](https://tapa.readthedocs.io/en/release/overview/overview.html), [Dynamatic](https://dynamatic.epfl.ch/), and [HeteroCL](https://heterocl.csl.cornell.edu/). However, these tools are developed independently with different compilation flows, which brings difficulties of integrating them together. [MLIR](https://mlir.llvm.org/) is a new compiler design paradigm where the source language is compiled through multiple levels of modularized intermediate representations (IRs), also known as dialects. Dialects act like domain-specific languages (DSLs) and can capture the approprate details at each level of abstraction.

The [CIRCT](https://circt.llvm.org/) project expands the MLIR-based development methodology for hardware design. It represents key components of hardware as MLIR dialects such as finite state machines (FSM), pipelines, and interface handshaking. HeteroCL has been migrated to the MLIR ecosystem as a dialect, with a new Python frontend called Allo. Allo decouples the interactions between the algorithm, hardware optimizations, and backend targets to enable productive design and testing. Lastly, Accelerator Memory Compiler (AMC) is an MLIR dialect for representing memory architecture. Its expressiveness is able to capture common memory organization strategies such as partitioning, banking, and arbitration. AMC can be further lowered to Calyx, which is also part of the CIRCT ecosystem. Calyx IR gives us a pathway to finally to synthesizable Verilog. The contribution of this project is that we integrated Allo with AMC to enable a Python frontend for AMC. This allows us to use Allo to describe the algorithm and AMC to describe the memory architecture. The resulting design can be compiled to Verilog and simulated with a single function call. In the end, we hope that this integration will enable a more productive design flow for hardware accelerators as well as help us find more bugs in AMC.

## Design Example

To use Allo with AMC, the designer only needs to write their kernel in Python. Then, then user can simply specify which Python function to build with the AMC backend. Moreover, AMC acts as a drop-in replacement to the other backends in the Allo ecosystem, making functional testing and debugging seamless. In the far majority of cases, the [NumPy](https://numpy.org/) or the LLVM backend is suitable for use as a golden reference model. In this section, we walk through an example where we functionally verify a kernel built with AMC. Then, we will record some resource estimates and execution times.

Our illustrative example will be matrix multiplication. What would ordinarily be a cumbersome task when using the vendor tools, like [Vitis HLS](https://www.xilinx.com/products/design-tools/vitis/vitis-hls.html), becomes a simple, 5 minute exercise. First, we specify some inputs initialized to random values. Then, `build()` the accelerator. Finally, we call both the software and hardware simulations and check their outputs. Compared to a C/C++ based tool flow, the amount of boilerplate code and scripting is reduced to near zero. In the end, we can represent this application with only 18 lines of code:

```python
def test_amc():
    N = 16
    # Initialize 2 input matricies to random integers
    A = np.random.randint(0, 20, size=(N, N), dtype="int32")
    B = np.random.randint(0, 20, size=(N, N), dtype="int32")

    # Our kernel is just matrix multiplication, Allo provides a primitive for this
    def kernel(A: int32[N, N], B: int32[N, N]) -> int32[N, N]:
        return allo.matmul(A, B)

    # Build the accelerator with AMC backend
    s = allo.customize(kernel)
    f = s.build(target="amc")
    # Run the software simulation by invoking directly
    np_out = kernel(A, B)
    # Now run the hardware simulation with AMC
    allo_out = f(A, B)
    np.testing.assert_array_equal(allo_out, np_out)
```

Additionally, we can also get an approximation of how much FPGA resources this design uses. Simply call `.get_resource_estimates()` after building:

```python
    print(f.get_resource_estimates())
    # {
    #   "BRAM36": 1,
    #   "DSP": 3,
    #   "FF": 255,
    #   "LUT": 467,
    #   "LUTL": 467,
    #   "LUTM": 0,
    #   "cycles": 15016,
    #   "name": "kernel"
    # }
```

This uses [Vivado](https://www.xilinx.com/products/design-tools/vivado.html) RTL synthesis for Xilinx FPGAs, but it would only be a one-time effort to support other synthesis tools like [Yosys](https://yosyshq.net/yosys/). If functional errors arise, the AMC backend also offers some options to help with debugging. To better understand what is going on under the hood, the first step would be to dump the underlying intermediate representation in MLIR with `print(f.module)`.

```mlir
// print(f.module)
func.func @kernel(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>) -> memref<16x16xi32> {
  %c0_i32 = arith.constant 0 : i32
  %alloc = memref.alloc() : memref<16x16xi32>
  affine.for %arg2 = 0 to 16 {
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 0 to 16 {
        %0 = affine.load %arg0[%arg2, %arg4] : memref<16x16xi32>
        %1 = affine.load %arg1[%arg4, %arg3] : memref<16x16xi32>
        %2 = affine.load %alloc[%arg2, %arg3] : memref<16x16xi32>
        %3 = arith.muli %0, %1 : i32
        %4 = arith.addi %2, %3 : i32
        affine.store %4, %alloc[%arg2, %arg3] : memref<16x16xi32>
      }
    }
  }
  return %alloc : memref<16x16xi32>
}
```

The above IR helps give insight into what program the AMC is actually attempting to compile. To debug at a lower level, we can inspect the IR at individual steps of the pass pipeline. For example, here is what this application would look like after AMC buffers are inserted:

```mlir
module {
  amc.memory @amcMemory0(!amc.static_port<16x16xi32, w, 1>, !amc.static_port<16x16xi32, rw, 1>) {
    %0 = amc.alloc : !amc.memref<16x16xi32>
    %1 = amc.create_port(%0 : !amc.memref<16x16xi32>) : !amc.static_port<16x16xi32, w, 1>
    %2 = amc.create_port(%0 : !amc.memref<16x16xi32>) : !amc.static_port<16x16xi32, rw, 1>
    amc.extern %1, %2 : !amc.static_port<16x16xi32, w, 1>, !amc.static_port<16x16xi32, rw, 1>
  }
  func.func @kernel(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>, %arg2: memref<16x16xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %0:2 = amc.inst @amcMemory0_inst of @amcMemory0
            : !amc.static_port<16x16xi32, w, 1>, !amc.static_port<16x16xi32, rw, 1>
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 0 to 16 {
        affine.for %arg5 = 0 to 16 {
          %1 = affine.load %arg0[%arg3, %arg5] : memref<16x16xi32>
          %2 = affine.load %arg1[%arg5, %arg4] : memref<16x16xi32>
          %3 = amc.affine_load %0#1[%arg3, %arg4] : !amc.static_port<16x16xi32, rw, 1>
          %4 = arith.muli %1, %2 : i32
          %5 = arith.addi %3, %4 : i32
          amc.affine_store %5, %0#1[%arg3, %arg4] : !amc.static_port<16x16xi32, rw, 1>
        } {hls.pipeline, hls.unroll = 1 : i64}
      } {hls.unroll = 1 : i64}
    } {hls.unroll = 1 : i64}
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 0 to 16 {
        %1 = amc.affine_load %0#1[%arg3, %arg4] : !amc.static_port<16x16xi32, rw, 1>
        affine.store %1, %arg2[%arg3, %arg4] : memref<16x16xi32>
      } {hls.bufferize, hls.pipeline, hls.unroll = 1 : i64}
    } {hls.unroll = 1 : i64}
    return
  }
}
```

For more information on the motivation behind AMC and the dialect itself, you can visit the [Andrew Butt's blog post](https://www.cs.cornell.edu/courses/cs6120/2022sp/blog/banked-memory-compiler-backend-in-mlir/) from last year. As a last resort when debugging, the AMC backend has a `.dump_vcd()` method which outputs the waveforms from the hardware simulation:

```python
    f.dump_vcd("waveform.vcd")
    # You can now take the debugging to your favorite waveform viewer
```

Writing other types of kernels is as easy as writing normal Python. For example, here are a couple of other kernels which are just a simple to get up and running:

```python
def fibonnaci(A: uint32[N]):
    A[0] = 1
    A[1] = 1
    for i in range(2, N):
        A[i] = A[i - 1] + A[i - 2]

def vadd(A: uint32[N], B: uint32[N]) -> uint32[N]:
    return A + B
```

Outside of what is shown here, the Allo DSL allows much more elaborate kernels and control over compute customizations, like loop tiling and reuse buffers. One again, you can get more information on Allo and its MLIR dialect by reading last years blog post by [Hongzheng Chen](https://www.cs.cornell.edu/courses/cs6120/2022sp/blog/hcl-mlir/).

To conclude, we hope this example demonstrates the usefulness of the Allo frontend for high-level hardware design and further development of the AMC HLS compiler. As far as we are aware, the only other frameworks using Python to drive FPGA high-level synthesis are [LeFlow](https://arxiv.org/abs/1807.05317) and [PyLog](https://ieeexplore.ieee.org/document/9591456). However, neither of these efforts are using a homebrewed HLS compiler like us.

## Tool flow

Under the hood, the Allo frontend is automating all the interactions with other tools, IRs, and frameworks. Nonetheless, understanding the interactions of each component is important to understanding the novelty in our approach.

<center>
<img src="allo_dependencies.png" alt="Diagram of build dependencies" title="Allo build dependencies" style="zoom:55%;">
</center>

The top row of dependencies are C++ codebases linked together as static libraries. We enter interact with this library with input MLIR generated from Allo. On the backend, we emit Calyx which is lowered to Verilog by a separate [Calyx compiler](https://github.com/cucapra/calyx) that is written in Rust.

The HCL dialect and passes are primarily responsible for processing the customization directives of the Allo frontend. Then, AMC and CIRCT carry out the "traditional" steps of high-level synthesis: allocation, scheduling, and binding. Finally, the Calyx compiler generates the data paths and control logic for the scheduled program.

### Overview

### Allo
Allo leverages an algorithm-optimization decoupled paradigm, which means users can first define the algorithm in a high-level language and then optimize the program with various hardware customization techniques (i.e., schedule primitives). Back to the matmul example, without using the provided primitive, the code would look like this:

```python
def matmul(A: int32[16, 16], B: int32[16, 16]) -> int32[16, 16]:
    C: int32[16, 16] = 0
    for i, j, k in allo.grid(16, 16, 16):
        C[i, j] += A[i, k] * B[k, j]
    return C
```

[This blog post](https://siboehm.com/articles/22/Fast-MMM-on-CPU) describes an effective way of optimizing matrix multiplication. The key idea is to reorder and tile the loops for a better data locality. The optimized loop order would be `i, k, j`. With Allo, we can easily achieve this with just two lines of code:

```python
schedule = allo.customize(matmul)
schedule.reorder("i", "k", "j")
```

Loop tiling is supported by the `split()` directive:
```python
schedule.split("i", factor=4)
```

Finally, we can build the design for various backends:
```python
module = schedule.build(target="llvm") # can also be 'vlhs', 'amc'
```

TODO: what is the most important feature of Allo for this project?

### AMC

AMC (Accelerator Memory Compiler) is an entirely new plane of intermediate representation dedicated to representing memory architecture. To explain the need for such a dialect, the current state of conventional high-level synthesis compilers must be understood. As a brief summary, current HLS tools lacks an expressive model of memory, and it takes great manual effort to unlock the power of spatial architectures. This is by and large a consequence of the chosen source language. Most HLS compilers have a C/C++ frontend and LLVM middle end. As a consequence, every HLS compiler also has a set of compiler directives (`#pragma`'s) to fill in the semantic gaps of compiling to spatial hardware. For example, there are pragmas for partitioning memories, instantiating FIFOs, loop pipelining, and much more. By not having first class constructs for these design elements, optimization becomes tightly-coupled with source rewrites and HLS falls flat on its promises of high productivity. For the interested reader, we think this [blog post](https://specbranch.com/posts/fpgas-what-happened/) helps explain the current state of the FPGA accelerators for outsiders.

Back to AMC, the custom dialect elaborates the *real* limiting resources of memory on spatial architectures: the ports. Each embedded block RAM on the FPGA has only two ports, and it takes an intelligent design methodology to utilize these memories in such a way that maximizes performance. AMC eliminates the guessing game by elaborating the description of the memory subsystem and optimizing it for the program at hand. To accompany the AMC dialect, we use [Calyx IR](https://calyxir.org/) to represent the control logic of the scheduled program. Here is a very simple program allocated with an AMC memory:

```mlir
  amc.memory @amcMemory0(!amc.dynamic_port<256xi32, w>, !amc.dynamic_port<128xi32, r>, !amc.dynamic_port<128xi32, r>) {
    %0 = amc.alloc : !amc.memref<256xi32>
    %1 = amc.create_port(%0 : !amc.memref<256xi32>) : !amc.dynamic_port<256xi32, w>
    %2 = amc.create_port(%0 : !amc.memref<256xi32>) : !amc.dynamic_port<256xi32, r>
    amc.extern %1, %2, %2 : !amc.dynamic_port<256xi32, w>, !amc.dynamic_port<256xi32, r>, !amc.dynamic_port<256xi32, r>
  }
  func.func @main(%arg0: memref<256xi32>, %arg1: memref<256xi32>, %arg2: memref<256xi32>) {
    %0:3 = amc.inst @amcMemory0_inst of @amcMemory0 : !amc.dynamic_port<256xi32, w>, !amc.dynamic_port<256xi32, r>,
           !amc.dynamic_port<256xi32, r>
    // Buffer data into a scratchpad
    affine.for %arg3 = 0 to 256 {
      %1 = affine.load %arg0[%arg3] : memref<256xi32>
      amc.affine_store %1, %0#0[%arg3] : !amc.dynamic_port<256xi32, w>
    }
    // Split data into two separate outputs
    affine.for %arg3 = 0 to 128 {
      %1 = amc.affine_load %0#1[%arg3] : !amc.dynamic_port<256xi32, r>
      %2 = amc.affine_load %0#2[%arg3 + 128] : !amc.dynamic_port<256xi32, r>
      affine.store %1, %arg1[%arg3] : memref<128xi32>
      affine.store %2, %arg2[%arg3 + 128] : memref<128xi32>
    }
    return
  }
```

The following diagram shows the full pass pipeline of compiling the core MLIR dialects to Verilog:

<center>
<img src="amc_passes.png" alt="Diagram for AMC pass pipeline" title="AMC pass pipeline" style="zoom:25%;">
</center>

## Results

In this section, we report the latency and resource measures of a select set of micro-benchmarks. By using small testcases, we have the best chance of understanding how the high-level constructs are being mapped to hardware and where the compiler inefficiencies lie. With that said, here are a table of benchmarks compiled with our toolflow versus Vitis C HLS.

**AMC**
| Benchmark     | Latency (Cycles) | LUTs | FFs | BRAM36s | DSPs |
| ------------- | ---------------- | ---- | --- | ------- | ---- |
| matmul16x16   | 15016            | 467  | 255 | 1       | 3    |
| spmv20x20     | 48               | 183  | 309 | 0       | 3    |
| vadd20        | 112              | 612  | 305 | 2       | 0    |
| fibonacci20   | 77               | 120  | 151 | 0       | 0    |

**Vitis 2022**
| Benchmark     | Latency (Cycles) | LUTs | FFs | BRAM36s | DSPs |
| ------------- | ---------------- | ---- | --- | ------- | ---- |
| matmul16x16   | 5409             | 221  | 74  | 0       | 3    |
| spmv20x20     | 42               | 249  | 145 | 0       | 3    |
| vadd20        | 22               | 81   | 13  | 0       | 3    |
| fibonacci20   | 41               | 226  | 50  | 0       | 0    |

**Difference: AMC - Vitis**
| Benchmark     | Latency (Cycles) | LUTs  | FFs    | BRAM36s | DSPs  |
| ------------- | ---------------- | ----- | ------ | ------- | ----- |
| matmul16x16   | +170%            | +110% | +240%  | -       | +0%   |
| spmv20x20     | +14%             | -26%  | +110%  | -       | +0%   |
| vadd20        | +410%            | +760% | +2200% | -       | -300% |
| fibonacci20   | +88%             | -47%  | +200%  | -       | -     |

The main story here is revealed when looking at the core MLIR dialects Allo is emitting after parsing the AST. Inefficiencies in how Allo infers data types and creates `affine.for` loop nests is creating way too many redundant memory operations. For example, Allo is not using store-to-load forwarding between loop iterations. Even worse, Allo tries to explicity infer data type conversions when using operations that change the data width (e.g. 32b + 32b = 33b). This for some reason is causing extra memory copy loops to convert an entire memory before it is used. We anticipate that fixing the Allo frontend to produce higher-quality loops will take some time, as it depends on some level of memory dependence analysis. Nonetheless, it will be a very important improvement to make in order to reach similar latencies as what Vitis can produce with C code.

## Future Work

Given that the focus of this project was primarily the software plumbing required to get things working end-to-end, there is still a lot of work needed to increase the quality of results. Right now, there are too many ways to accidentally create a slow design that has many redundant memory operations. Moreover, some designs just don't work, exposing bugs in our scheduling pass. Here is a list of features and fixes we intend to eventually implement:

- Fix how Allo constructs `affine.for` loops. Right now Allo is not using inter-iteration arguments, meaning the data flow between loop iterations is through memory reads and writes. This is much slower, because it artificially increases the II of the pipeline.
- Handle scalar values. Allo creates a memref cell to hold scalar values, and this once again limits performance. Every interaction with the variable is with loads and stores, instead of doing the proper SSA conversion.
- Fix scheduler bugs.
- Use Allo customization directives to assist AMC in inferring more efficient scratchpad memories. For example, array partitioning and memory access patterns are hints that Allo could provide to our allocation pass.

## Conclusion

The project was by and large a success, because we have achieved a much higher level of automation in evaluating the AMC+Calyx toolflow. Being able to write HLS kernels with `numpy` and run the RTL simulation as a normal function call greatly reduces the amount of effort in adding a new test case. Moreover, the `.dump_vcd()` and `.get_resource_estimates()` provide more tools for debugging without having to manually interact with the synthesis tools.