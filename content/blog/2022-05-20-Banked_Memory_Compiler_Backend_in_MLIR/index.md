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

### Code

Unfortunately, as this work is related to ongoing research, we are not yet ready to open-source the project. We have shared the code with Adrian which is available [here](https://www.github.com). If anyone else in the class in interested in seeing the code please reach out and we will give you access as well.

### Introduction

Banked memories, or logical memories that are divided into multiple physical memories, are an important part of achieving high performance FPGA designs. Unfortunately, neither RTL- nor HLS-based languages provide sufficient abstractions to express banked memories. This results in tremendous efforts that programmers need to put when designing memory sub-systems for their hardware architectures. The problem is especially hard given the diverse set of applications with different memory access patterns and performance requirements, as well as numerous hardware primitives the memory can be map onto (e.g. LUTRAM, BRAM, ULTRARAM, etc.) available in today's FPGAs. Without these abstractions, designers have to manually plan the microarchitecture of the memories with all the necessary optimizations in order to make them meeting target design requirements.

To address the aforementioned challenges and simplify the process of designing banked memories, we propose AMC -- a novel MLIR dialect built on top of CIRCT and Calyx that enables compilation of arbitrary memory structures. Our contribution lies in the following three aspects:

* the AMC dialect -- a programming IR abstraction for expressing banked memories;
* the AMC compiler -- a reference implementation of the AMC dialect built on Calyx;
* an optimization pass showcasing how our dialect can enable implementation of efficient memory structures when lowering down to hardware.


### The AMC dialect

We build the AMC dialect on top of Calyx. Calyx is an IR for compilers generating hardware accelerators. The banked memory specification in the AMC dialect involves four parts: (1) interface specification, (2) memory allocation specification, (3) port specification, and (4) port-interface mapping.

*The memory interface* specifies all read and write interfaces to a unit of banked memory with the corresponding set of arguments. The arguments include the address space size, data width, read/write semantics, and memory access latency in cycles. Example of an interface definition is shown in the listing bellow. Here we define a memory component *@test1* that implements a banked memory with four ports (two on read, two on write), each having 512 32-bit words and 1 cycle of guaranteed access latency.
```mlir
amc.memory @test1(!amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>, !amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>);
```

*The memory allocation description* specifies how the memory is divided into banks. For example, ```%0 = amc.alloc : !amc.memref<1024xf32, bank [2]>``` says that the memory contains two equivalent banks with 1024 32-bit words in total, and it uses \%0 as a reference.

*Port specification* allows to assign read/write ports to the memory banks described earlier. For example,
```mlir
%3 = amc.create_port(%0 : !amc.memref<1024xf32, bank [2]>) banks [1] : !amc.port<512xf32, r, 1>
```
describes a read port connected to the bank 2 of the memory referred by \%0. Here, port attributes have the same meaning as in the interface description.

Finally, *port-interface mapping* specifies how exactly the bank ports are mapped onto the memory interface ports defined earlier. A completed example of possible banked memory definition with our AMC dialect is shown in the listing bellow:
```mlir
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
### Lowering

The general lowering pipeline can be view as follows:

Handshake Lowering -> Merge Lowering -> Bank Lowering -> Optimization -> AmcToCalyx

Our general goal here is to perform the straight forward lowering passes first, and then optimize later one we have fully expanded the memory structure. We will dive into each of these lowering passes below.

### Bank Lowering

We will start with the simplest lowering step. This should be viewed as the final lowering pass pre-optimization and simply translated a banked memref into multiple unbanked memref. This step requires that all create ports only access a single bank and create only latency-sensitive ports.

Here is an example of bank lowering:
```mlir
// Input
amc.memory @test1(!amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>, !amc.port<512xf32, r, 1>, !amc.port<512xf32,
  %0 = amc.alloc : !amc.memref<1024xf32, bank [2]>
  %1 = amc.create_port(%0 : !amc.memref<1024xf32, bank [2]>) banks [0] : !amc.port<512xf32, r, 1>
  %2 = amc.create_port(%0 : !amc.memref<1024xf32, bank [2]>) banks [0] : !amc.port<512xf32, w, 1>
  %3 = amc.create_port(%0 : !amc.memref<1024xf32, bank [2]>) banks [1] : !amc.port<512xf32, r, 1>
  %4 = amc.create_port(%0 : !amc.memref<1024xf32, bank [2]>) banks [1] : !amc.port<512xf32, w, 1>
   
  amc.extern %1, %2, %3, %4 : !amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>, !amc.port<512xf32, r, 1>, !amc.p
}

// Output after bank lowering
amc.memory @test1(!amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>, !amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>) {
  %0 = amc.alloc : !amc.memref<512xf32>
  %1 = amc.alloc : !amc.memref<512xf32>
  %2 = amc.create_port(%0 : !amc.memref<512xf32>) : !amc.port<512xf32, r, 1>
  %3 = amc.create_port(%0 : !amc.memref<512xf32>) : !amc.port<512xf32, w, 1>
  %4 = amc.create_port(%1 : !amc.memref<512xf32>) : !amc.port<512xf32, r, 1>
  %5 = amc.create_port(%1 : !amc.memref<512xf32>) : !amc.port<512xf32, w, 1>
  amc.extern %2, %3, %4, %5 : !amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>, !amc.port<512xf32, r, 1>, !amc.port<512xf32, w, 1>
}
```

### Merge Lowering

The purpose of merge lowering is to translate latency-sensitive port creations that access more than one bank on a memory into multiple ports that access individual banks. These smaller ports are then combined with the merge operator to produce a port of the original size. This is analogous to how a port of this type would actually be implemented in hardware and allows future lowering and optimization passes to have a better idea of how many ports are actually accessing each bank.

The merge operator is a new primitive that takes multiple latency-sensitive ports and produces one larger latency-sensitive port. This can be thought of as a type of mux, which based on the top ceil(log_2(n)) bits of the address will select one of the n ports to forward the access to.

Here is an example of merge lowering:
```mlir
// Input
amc.memory @test2(!amc.port<512xf32, rw, 1>, !amc.port<512xf32, r, 1>, !amc.port<1024xf32, w, 1>) {   
  %0 = amc.alloc : !amc.memref<1024xf32, bank [2]>   
  %1 = amc.create_port(%0 : !amc.memref<1024xf32, bank [2]>) banks [0] : !amc.port<512xf32, r, 1>   
  %2 = amc.create_port(%0 : !amc.memref<1024xf32, bank [2]>) banks [1] : !amc.port<512xf32, r, 1>   
  %3 = amc.create_port(%0 : !amc.memref<1024xf32, bank [2]>) banks [0, 1] : !amc.port<1024xf32, w, 1>   

  amc.extern %1, %2, %3 : !amc.port<512xf32, r, 1>, !amc.port<512xf32, r, 1>, !amc.port<1024xf32, w, 1>   
}

// Output after merge lowering
amc.memory @test2(!amc.port<512xf32, rw, 1>, !amc.port<512xf32, r, 1>, !amc.port<1024xf32, w, 1>) {
  %0 = amc.alloc : !amc.memref<1024xf32, [2]>
  %1 = amc.create_port(%0 : !amc.memref<1024xf32, [2]>) banks [0] : !amc.port<512xf32, r, 1>
  %2 = amc.create_port(%0 : !amc.memref<1024xf32, [2]>) banks [1] : !amc.port<512xf32, r, 1>
  %3 = amc.create_port(%0 : !amc.memref<1024xf32, [2]>) banks [0] : !amc.port<512xf32, w, 1>
  %4 = amc.create_port(%0 : !amc.memref<1024xf32, [2]>) banks [1] : !amc.port<512xf32, w, 1>
  %5 = amc.merge(%3, %4 : !amc.port<512xf32, w, 1>, !amc.port<512xf32, w, 1>) : !amc.port<1024xf32, w, 1>
  amc.extern %1, %2, %5 : !amc.port<512xf32, r, 1>, !amc.port<512xf32, r, 1>, !amc.port<1024xf32, w, 1>
}

// Output after merge lowering and bank lowering
amc.memory @test2(!amc.port<512xf32, rw, 1>, !amc.port<512xf32, r, 1>, !amc.port<1024xf32, w, 1>) {
  %0 = amc.alloc : !amc.memref<512xf32>
  %1 = amc.alloc : !amc.memref<512xf32>
  %2 = amc.create_port(%0 : !amc.memref<512xf32>) : !amc.port<512xf32, r, 1>
  %3 = amc.create_port(%1 : !amc.memref<512xf32>) : !amc.port<512xf32, r, 1>
  %4 = amc.create_port(%0 : !amc.memref<512xf32>) : !amc.port<512xf32, w, 1>
  %5 = amc.create_port(%1 : !amc.memref<512xf32>) : !amc.port<512xf32, w, 1>
  %6 = amc.merge(%4, %5 : !amc.port<512xf32, w, 1>, !amc.port<512xf32, w, 1>) : !amc.port<1024xf32, w, 1>
  amc.extern %2, %3, %6 : !amc.port<512xf32, r, 1>, !amc.port<512xf32, r, 1>, !amc.port<1024xf32, w, 1>
}
```

### Handshake Lowering

Throughout the design of the AMC dialect, we discovered that it is important to distinguish between latency-sensitive and latency-insensitive (handshake) ports. Unlike latency-sensitive ports that provide a fixed latency between a read request and data availability, handshake ports must wait for a valid signal before data is available. Handshake ports are useful for irregular paralleism, where the access patterns are not known at compile time. This allows for memories that can provide a consistent latency in most cases, but stall one or more accesses in the case of a bank conflict.

Hanshake ports must be lowered before being fed into the rest of the lowering pipeline. Currently, we lower handshake ports in the most naive way possible, assuming that a future optimization pass will combine arbiters in a sensible way. This pass has not yet been implemented.

To support handshake ports we must introduce a new primitive, the arbiter. The job of an arbiter is to take in some number of latency-senstive port ssa value and produce some number of latency-insensitive handshake port ssa values. These handshake port ssa values can then be passed with amc.extern to the top level IO. At this level, the job of the arbiter is to prevent multiple handshake ports from trying to access the same latency-sensitive port at the same time. We will later have to choose a specific arbitration scheme (fixed-priority, round-robin, etc.) to meet the needs to the memory interface.

Here is an example of handshake lowering:
```mlir
// Input
amc.memory @test2(!amc.port_hs<1024xi32, rw>, !amc.port_hs<1024xi32, rw>) {
  %0 = amc.alloc : !amc.memref<1024xi32, [2]>
  %1 = amc.create_port(%0 : !amc.memref<1024xi32, [2]>) banks [0, 1] : !amc.port_hs<1024xi32, rw>
  %2 = amc.create_port(%0 : !amc.memref<1024xi32, [2]>) banks [0, 1] : !amc.port_hs<1024xi32, rw>
  amc.extern %1, %2 : !amc.port_hs<1024xi32, rw>, !amc.port_hs<1024xi32, rw>
}

// Output after only handshake lowering
amc.memory @test2(!amc.port_hs<1024xi32, rw>, !amc.port_hs<1024xi32, rw>) {
  %0 = amc.alloc : !amc.memref<1024xi32, [2]>
  %1 = amc.create_port(%0 : !amc.memref<1024xi32, [2]>) banks [0, 1] : !amc.port<1024xi32, rw, 1>
  %2 = amc.arbiter(%1 : !amc.port<1024xi32, rw, 1>) banks [0, 1] : !amc.port_hs<1024xi32, rw>
  %3 = amc.create_port(%0 : !amc.memref<1024xi32, [2]>) banks [0, 1] : !amc.port<1024xi32, rw, 1>
  %4 = amc.arbiter(%3 : !amc.port<1024xi32, rw, 1>) banks [0, 1] : !amc.port_hs<1024xi32, rw>
  amc.extern %2, %4 : !amc.port_hs<1024xi32, rw>, !amc.port_hs<1024xi32, rw>
}

// Output after handshake lowering and merge lowering
amc.memory @test2(!amc.port_hs<1024xi32, rw>, !amc.port_hs<1024xi32, rw>) {
  %0 = amc.alloc : !amc.memref<1024xi32, [2]>
  %1 = amc.create_port(%0 : !amc.memref<1024xi32, [2]>) banks [0] : !amc.port<512xi32, rw, 1>
  %2 = amc.create_port(%0 : !amc.memref<1024xi32, [2]>) banks [1] : !amc.port<512xi32, rw, 1>
  %3 = amc.merge(%1, %2 : !amc.port<512xi32, rw, 1>, !amc.port<512xi32, rw, 1>) : !amc.port<1024xi32, rw, 1>
  %4 = amc.arbiter(%3 : !amc.port<1024xi32, rw, 1>) banks [0, 1] : !amc.port_hs<1024xi32, rw>
  %5 = amc.create_port(%0 : !amc.memref<1024xi32, [2]>) banks [0] : !amc.port<512xi32, rw, 1>
  %6 = amc.create_port(%0 : !amc.memref<1024xi32, [2]>) banks [1] : !amc.port<512xi32, rw, 1>
  %7 = amc.merge(%5, %6 : !amc.port<512xi32, rw, 1>, !amc.port<512xi32, rw, 1>) : !amc.port<1024xi32, rw, 1>
  %8 = amc.arbiter(%7 : !amc.port<1024xi32, rw, 1>) banks [0, 1] : !amc.port_hs<1024xi32, rw>
  amc.extern %4, %8 : !amc.port_hs<1024xi32, rw>, !amc.port_hs<1024xi32, rw>
}

// Output after handshake lowering, merge lowering, and bank lowering
amc.memory @test2(!amc.port_hs<1024xi32, rw>, !amc.port_hs<1024xi32, rw>) {
  %0 = amc.alloc : !amc.memref<512xi32>
  %1 = amc.alloc : !amc.memref<512xi32>
  %2 = amc.create_port(%0 : !amc.memref<512xi32>) : !amc.port<512xi32, rw, 1>
  %3 = amc.create_port(%1 : !amc.memref<512xi32>) : !amc.port<512xi32, rw, 1>
  %4 = amc.merge(%2, %3 : !amc.port<512xi32, rw, 1>, !amc.port<512xi32, rw, 1>) : !amc.port<1024xi32, rw, 1>
  %5 = amc.arbiter(%4 : !amc.port<1024xi32, rw, 1>) banks [0, 1] : !amc.port_hs<1024xi32, rw>
  %6 = amc.create_port(%0 : !amc.memref<512xi32>) : !amc.port<512xi32, rw, 1>
  %7 = amc.create_port(%1 : !amc.memref<512xi32>) : !amc.port<512xi32, rw, 1>
  %8 = amc.merge(%6, %7 : !amc.port<512xi32, rw, 1>, !amc.port<512xi32, rw, 1>) : !amc.port<1024xi32, rw, 1>
  %9 = amc.arbiter(%8 : !amc.port<1024xi32, rw, 1>) banks [0, 1] : !amc.port_hs<1024xi32, rw>
  amc.extern %5, %9 : !amc.port_hs<1024xi32, rw>, !amc.port_hs<1024xi32, rw>
}
```

Here we can see what is meant by lowering in the most naive way. Each handshake port has its own arbiter which functionally just ties the valid signal high as each handshake port can always access the underlying latency-sensitive port. A future pass could co-optimize the underlying memory primitives and the handshake ports by merging together some arbiters to reduce the number of underlying ports.

### Optimization Pass: Memory Aggregation

After handshake, merge, and bank lowering we have a consistent way to optimize the underlying memory banks. To demonstrate optimization potentials, we implemented s memory aggregation pass. This pass allows to reduce the memory depth in favor of the word length. For example, a memory of type ```!amc.memref<1024xi32>``` can be converted to ```!amc.memref<512xi64>```. This optimization can be useful when it helps to better pack memory in the available hardware units. In particular, if the FPGAs ultra-RAM units have the width of 72 bits, it might be beneficial to aggregate memory as we showed in the above example.

The aggregation pass consists of four steps. The pass first replaces the type of memory allocation with another memory type of reduced depth. It then fixes all references to that memory for the `create_port` operations for all ports that use this memory. Then the pass injects a new operation `split_aggregate` that transforms ports of aggregated types into the ports of the original types as shown in the example bellow:
```mlir
%1 = amc.create_port(%0 : !amc.memref<256xi64>) : !amc.port<256xi64, w, 1>
%2 = amc.split_aggregated(%1 : !amc.port<256xi64, w, 1>) : !amc.port<512xf32, w, 1>
```

`split_aggregate` needs to be synthesized into a hardware circuit that implements the following functionality. It takes `N - 1` LSB of the original N-wide address and looks-up memory by this address. The resulting read/write word has the double width of the data port as in the memory specification. Then depending on the `N'th` LSB of the original address, either the LSB of the MSB part of the accessed word is getting assigned to the I/O ports.

The listing bellow shows example of the banked memory specification when applying the optimization.

```mlir
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

### Lowering AMC Dialect down to Calyx

### Challenges

### Evaluation

### Future Work

One of the biggest limitations we faced when lowering AMC dialect to Calyx is the limited support of certain memory operations in Calyx itself. In particular, Calyx lacks support for multi-ported memories, memories with a specific access latency, and memories that will map to a specific FPGA primitive. These should be able to be added using an external Calyx library, but this is not currently supported by by the Calyx MLIR dialect which we are lowering to.

As the future work we are planning to extend Calyx to support these operations so we can implement more functionality in AMC dialect and showcase more benefits of it over existing HLS tools. The lowering pass will also need to be more robust to support a wide range of optimizations and the memory interface will need to be interfaced with a scheduler of some sort to actual generate full designs. Comprehensive evaluation of AMC dialect and implementation of more optimizations is another major part of the future work.
