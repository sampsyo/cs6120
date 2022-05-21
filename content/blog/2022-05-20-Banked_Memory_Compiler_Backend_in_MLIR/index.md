+++
title = "Banked Memory Compiler Backend in MLIR"
[extra]
bio = """
  [Andrew Butt](https://andrewb1999.github.io/) is a first-year PhD student at Cornell's ECE department;
  [Nikita Lazarev](https://www.nikita.tech/) is a third-year PhD student at Cornell's ECE department;
"""
[[extra.authors]]
name = "Andrew Butt"
link = "https://andrewb1999.github.io/"  # Links are optional.
[[extra.authors]]
name = "Nikita Lazarev"
link = "https://www.nikita.tech/"  # Links are optional.
+++


### Introduction

Banked memories, or logical memories that are divided into multiple physical memories, are an important part of achieving high performance FPGA designs. Unfortunately, neither RTL- not HLS-based languages provide sufficient abstractions to express banked memories. This results in tremendous efforts that programmers need to put when designing memory sub-systems for their hardware architectures. The problem is especially hard given the diverse set of applications with different memory access patterns and performance requirements, as well as numerous hardware primitives the memory can be map onto (e.g. LUTRAM, BRAM, ULTRARAM, etc.) available in today's FPGAs. Without these abstractions, designers have to manually plan the microarchitecture of the memories with all the necessary optimizations in order to make them meeting target design requirements.

To address the aforementioned challenges and simplify the process of designing banked memories, we propose AMC -- a novel MLIR dialect built on top of CIRCT and Calyx that enables compilation of arbitrary memory structures. Our contribution lies in the following three aspects:

* the AMC dialect -- a programming IR abstraction for expressing banked memories;
* the AMC compiler -- a reference implementation of the AMC dialect built on Calyx;
* an optimization pass showcasing how our dialect can enable implementation of efficient memory structures when lowering down to hardware.


### The AMC dialect

We build the AMC dialect on top of Calyx. Calyx is an IR for compilers generating hardware accelerators. Banked memory specification in our AMC dialect involves four parts: (1) interface specification, (2) memory allocation specification, (3) port specification, and (4) port-interface mapping.

*The memory interface* specifies all read and write interfaces to a unit of banked memory with the corresponding set of arguments. The arguments include the address space size, data width, read/write semantics, and memory access latency in cycles. Example of an interface definition is shown in the listing bellow. Here we define a memory component *@test1* that implements a banked memory with four ports (two on read, two on write), each having 512 32-bit words and 1 cycle of guaranteed access latency.
```
amc.memory @test1(!amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>, !amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>);
```

*The memory allocation description* specifies how the memory is divided into banks. For example, ```%0 = amc.alloc : !amc.memref<1024xf32, bank [2]>``` says that the memory contains two equivalent banks with 1024 32-bit words in total, and it uses \%0 as a reference.

*Port specification* allows to assign read/write ports to the memory banks described earlier. For example,
```
%3 = amc.create_port(%0 : !amc.memref<1024xf32, bank [2]>) banks [1] : !amc.port<512xf32, r, 1>
```
describes a read port connected to the bank 2 of the memory referred by \%0. Here, port attributes have the same meaning as in the interface description.

Finally, *port-interface mapping* specifies how exactly the bank ports are mapped onto the memory interface ports defined earlier. A completed example of possible banked memory definition with our AMC dialect is shown in the listing bellow:
```
amc.memory @test1(!amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>, !amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>) {
  // Specify memory allocation.
  %0 = amc.alloc : !amc.memref<1024xf32, bank [2]>

  // Specify ports.
  %1 = amc.create_port(%0 : !amc.memref<1024xf32, bank [2]>) banks [0] : !amc.port<512xf32, r, 1>
  %2 = amc.create_port(%0 : !amc.memref<1024xf32, bank [2]>) banks [0] : !amc.port<512xf32, w, 1>
  %3 = amc.create_port(%0 : !amc.memref<1024xf32, bank [2]>) banks [1] : !amc.port<512xf32, r, 1>
  %4 = amc.create_port(%0 : !amc.memref<1024xf32, bank [2]>) banks [1] : !amc.port<512xf32, w, 1>
 
  // Specify I/O mapping.
  amc.extern %1, %2, %3, %4 : !amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>, !amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>
}
```


### Lowering AMC Dialect down to Calyx

TODO: Andrew


### Optimization Pass: Memory Aggregation

We implemented the memory aggregation pass. This pass allows to reduce the memory depth in favor of the word length. For example, a memory of type ```!amc.memref<1024xf32>``` can be converted to ```!amc.memref<512xf64>```. This optimization can be useful when it helps to better pack memory in the available hardware units. In particular, if the FPGAs block RAM units have the width of 72 bits, it might be beneficial to aggregate memory as we showed in the above example.

The aggregation pass consists of four steps. The pass first replaces the type of memory allocation with another memory type of reduced depth. It then fixes all references to that memory for the `create_port` operations for all ports that use this memory. Then the pass injects a new operation `split_aggregate` that transforms ports of aggregated types into the ports of the original types as shown in the example bellow:
```
%1 = amc.create_port(%0 : !amc.memref<256xi64>) : !amc.port<256xi64, w, 1>
%2 = amc.split_aggregated(%1 : !amc.port<256xi64, w, 1>) : !amc.port<512xf32, w, 1>
```

`split_aggregate` needs to be synthesized into a hardware circuit that implements the following functionality. It takes `N - 1` LSB of the original N-wide address and looks-up memory by this address. The resulting read/write word has the double width of the data port as in the memory specification. Then depending on the `N'th` LSB of the original address, either the LSB of the MSB part of the accessed word is getting assigned to the I/O ports.

The listing bellow shows example of the banked memory specification when applying the optimization.

```
 // Original specification.
 amc.memory @test1(!amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>, !amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>) {
   %0 = amc.alloc : !amc.memref<1024xf32, bank [2]>
   %1 = amc.create_port(%0 : !amc.memref<1024xf32, bank [2]>) banks [0] : ! >
   %2 = amc.create_port(%0 : !amc.memref<1024xf32, bank [2]>) banks [0] : !amc.port<512xf32, w, 1>
   %3 = amc.create_port(%0 : !amc.memref<1024xf32, bank [2]>) banks [1] : !amc.port<512xf32, r, 1>
   %4 = amc.create_port(%0 : !amc.memref<1024xf32, bank [2]>) banks [1] : !amc.port<512xf32, w, 1>
   amc.extern %1, %2, %3, %4 : !amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>, !amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>
}

// Lowered and optimized specification.
amc.memory @test1(!amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>, !amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>) {
  %0 = amc.alloc : !amc.memref<256xi64>
  %1 = amc.alloc : !amc.memref<256xi64>
  %2 = amc.create_port(%0 : !amc.memref<256xi64>) : !amc.port<256xi64, r, 1>
  %3 = amc.split_aggregated(%2 : !amc.port<256xi64, r, 1>) : !amc.port<512xf32, r, 1>
  %4 = amc.create_port(%0 : !amc.memref<256xi64>) : !amc.port<256xi64, w, 1>
  %5 = amc.split_aggregated(%4 : !amc.port<256xi64, w, 1>) : !amc.port<512xf32, w, 1>
  %6 = amc.create_port(%1 : !amc.memref<256xi64>) : !amc.port<256xi64, r, 1>
  %7 = amc.split_aggregated(%6 : !amc.port<256xi64, r, 1>) : !amc.port<512xf32, r, 1>
  %8 = amc.create_port(%1 : !amc.memref<256xi64>) : !amc.port<256xi64, w, 1>
  %9 = amc.split_aggregated(%8 : !amc.port<256xi64, w, 1>) : !amc.port<512xf32, w, 1>
  amc.extern %3, %5, %7, %9 : !amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>, !amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>
}
```


### Future Work

One of the biggest limitations we faced when lowering AMC dialect to Calyx is the limited support of certain memory operations in Calyx itself. TODO: give an example.

As the future work we are planning to extend Calyx to support these operations so we can implement more functionality in AMC dialect and showcase more benefits of it over existing HLS tools. Comprehensive evaluation of AMC dialect and implementation of more optimizations is another major part of the future work.
