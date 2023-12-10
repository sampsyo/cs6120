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

### Custom Hardware Accelerators

TODO

### High Level Synthesis

TODO

### MLIR and Incubator Projects

TODO(introduce MLIR, CIRCT, hcl, amc dialects)

## Design Example

To use Allo with AMC, the designer only needs to write their kernel in Python. Then, then user can simply specify which Python function to build with the AMC backend. Moreover, AMC acts a drop-in replacement to the other backends in the Allo ecosystem, making functional testing and debugging seamless. In the far majority of cases, the [NumPy](https://numpy.org/) or the LLVM backend is suitable for use as a golden reference model. In this section, we walk through an example where we functionally verify a kernel built with AMC. Then, we will record some resource estimates and execution times. 

Our illustrative example will be matrix multiplication. What would ordinarily be a cumbersome task when using the vendor tools, like [Vitis HLS](https://www.xilinx.com/products/design-tools/vitis/vitis-hls.html), becomes a simple, 5 minute exercise. First, we specify some inputs initialized to random values. Then, `build()` the accelerator. Finally, we call both the software and hardware simulations and check their outputs. Compared to a C/C++ based tool flow, the amount of boilerplate code and scripting is reduced to near zero. In the end, we can represent this application with only 18 lines of code:

```python
def test_amc():
    N = 16
    # Initialize 2 input matricies to random integers
    A = np.random.randint(0, 20, size=(N, N), dtype="int32")
    B = np.random.randint(0, 20, size=(N, N), dtype="int32")

    # Our kernel is just matrix multiplication
    def kernel(A: int32[N, N], B: int32[N, N]) -> int32[N, N]:
        return matmul(A, B)

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
func.func @kernel(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>) -> memref<16x16xi32> attributes {itypes = "ss", otypes = "s", top} {
  %c0_i32 = arith.constant 0 : i32
  %alloc = memref.alloc() : memref<16x16xi32>
  affine.for %arg2 = 0 to 16 {
    affine.for %arg3 = 0 to 16 {
      affine.store %c0_i32, %alloc[%arg2, %arg3] : memref<16x16xi32>
    }
  }
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
  func.func @kernel(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>, %arg2: memref<16x16xi32>) attributes {itypes = "ss", otypes = "s", top} {
    %c0_i32 = arith.constant 0 : i32
    %0:2 = amc.inst @amcMemory0_inst of @amcMemory0 : !amc.static_port<16x16xi32, w, 1>, !amc.static_port<16x16xi32, rw, 1>
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 0 to 16 {
        amc.affine_store %c0_i32, %0#0[%arg3, %arg4] : !amc.static_port<16x16xi32, w, 1>
      } {hls.pipeline, hls.unroll = 1 : i64}
    } {hls.unroll = 1 : i64}
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

Writing other types of kernels is as easy as writing normal Python. For example, here are a handful of other kernels which are just a simple to get up and running:

```python
def fibonnaci(A: uint32[N]):
    A[1] = 1
    for i in range(2, N):
        A[i] = A[i - 1] + A[i - 2]

def vadd(A: uint32[N], B: uint32[N]) -> uint32[N]:
    return A + B
```

Outside of what is shown here, the Allo DSL allows much more elaborate kernels and control over compute customizations, like loop tiling and reuse buffers. One again, you can get more information on Allo and its MLIR dialect by reading last years blog post by [Hongzheng Chen](https://www.cs.cornell.edu/courses/cs6120/2022sp/blog/hcl-mlir/). 

To conclude, we hope this example demonstrates the usefulness of the Allo frontend for high-level hardware design and further development of the AMC HLS compiler. As far as we are aware, the only other frameworks using Python to drive FPGA high-level synthesis are [LeFlow](https://arxiv.org/abs/1807.05317) and [PyLog](https://ieeexplore.ieee.org/document/9591456). However, neither of these efforts are using a homebrewed HLS compiler like us.

## Tool flow

Under the hood, the Allo frontend is automating all the interactions with other tools, IRs, and frameworks. Nonetheless, understanding each component is important to understanding the novelty in our approach.

TODO(explain all the steps)

### Overview

### Allo

### AMC

TODO. Show pass pipeline

## Results

| Benchmark     | Latency (Cycles) | LUTs | FFs | BRAM36s | DSPs |
| ------------- | ---------------- | ---- | --- | ------- | ---- |
| matmul16x16   | 15016            | 467  | 255 | 1       | 3    |
| spmv20x20     | 48               | 183  | 309 | 0       | 3    |
| vadd20        | 112              | 612  | 305 | 2       | 0    |
| fibonacci20   | 77               | 120  | 151 | 0       | 0    |

## Future Work

TODO
- Fix how allow constructs affine for. Needs to Store to load forward

## Conclusion

The project was by and large a success, because we have achieved a much higher level of automation in evaluating the AMC+Calyx toolflow. Being able to write HLS kernels with `numpy` and run the RTL simulation as a normal function call greatly reduces the amount of effort in adding a new test case. Moreover, the `.dump_vcd()` and `.get_resource_estimates()` provide more tools for debugging without having to manually interact with the synthesis tools.