+++
title = "Chlorophyll: Synthesis-Aided Compiler for Low-Power Spatial Architectures"
[extra]
bio = """
  [Hongzheng Chen](https://chhzh123.github.io/) is a first-year CS PhD student at the Computer Systems Laboratory, Cornell University. His research interests include domain-specific languages, compiler optimization, and heterogeneous computing systems.
"""
[[extra.authors]]
name = "Hongzheng Chen"
link = "https://chhzh123.github.io/"
+++


**TL;DR**: This paper proposes *Chlorophyll*, the first synthesis-aided programming model and compiler for the low-power spatial architecture *GreenArray GA144*. It decomposes a complex compilation program into smaller synthesis subprograms -- partitioning, layout, code separation, and code generation. Experimental results show that Chlorophyll can significantly reduce the programming burden and achieve good performance for several benchmarks.

## Background

In this section, I will firstly introduce the background of the program synthesis and spatial architecture.

### Program Synthesis
Some examples we saw in [Lesson 13](https://www.cs.cornell.edu/courses/cs6120/2022sp/lesson/13/) are just a few cases of program synthesis. Based on the talk[^1] given by Prof. [James Bornholt](https://www.cs.utexas.edu/~bornholt/), we can roughly classify the application of program synthesis into three categories: approximate computing, black box systems, and hardware synthesis. For this paper, we will only focus on the last category -- hardware synthesis.

Traditionally, hardware programmers need to write circuits in [hardware description languages (HDL)](https://en.wikipedia.org/wiki/Hardware_description_language) like Verilog or VHLS (see the left figure below), and push the program through time-consuming logical synthesis and physical synthesis (placement and routing) to generate the desired hardware architecture. However, HDL is too low-level and lack modern language features, making the development process extremely long. Therefore, [high-level synthesis (HLS)](https://en.wikipedia.org/wiki/High-level_synthesis) is proposed to enable the programmers to write hardware circuits in a high-level language like C/C++. See the right figure below, programmers need to insert pragmas after the loops or the array declartion for HLS, which are quite similar to the OpenMP pragmas. HLS will then generate the corresponding circuit description in HDL based on the pragmas provided by the programmer. Leveraging HLS, programmers can enjoy the facilities of high-level languages, which greatly shorten the development time for hardware accelerators.

In general, synthesis is not a new technique for hardware compilation. It has already been widely used in nowadays circuits design. Some commercial examples include [Xilinx Vivado HLS](https://www.xilinx.com/support/documentation-navigation/design-hubs/dh0012-vivado-high-level-synthesis-hub.html) and [Intel FPGA oneAPI toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/fpga.html#gs.z1p9mh)).

![](hardware-synthesis.png)

### Spatial Architecture
A spatial architecture commonly has multiple identical processing elements (PEs) on the chip. Those PEs are connected with on-chip network, but the data transmission and network interconnection need to be manually configured. A typical example of spatial architecture nowadays is [systolic array](https://en.wikipedia.org/wiki/Systolic_array), which is the core component of Google's [TPU](https://cloud.google.com/tpu). Since deep neural networks are essentially doing matrix multiplication, using systolic array can maximally exploit the parallelism and greatly speedup the computation.

This paper targets another spatial architecture called GreenArray (GA144). It is a 18-bit stack-based processor with 144 cores dispatched on 8 rows and 18 columns. The blueprint is very similar to systolic array, but the interconnection is much more regular. Each core of GA144 has 2 circular stacks with less than 100 18-bit words private memory. It can only communicate using blocking read or write with its neighbor cores in the same row or column. Overall, it is very energy-efficient compared with general processors like CPU and GPU.

Though GA144 is low-power and environmentally-friendly, it uses a low-level stack-based language called [colorForth](http://www.euroforth.org/ef19/papers/oakford.pdf)/[arrayForth](http://www.greenarraychips.com/home/documents/greg/cf-intro.htm) to program. (Actually it is a very interesting language that uses color to express different semantics.) It requires programmers to write programs for each core and manually manipulate the stack for transfering the data. The intercommunication between cores also need to be manually configured, which places great burden on the programmer.

### Specturm of Computing Devices
The following figure shows the programmability and energy efficiency of different computing devices. Generally, CPU is the most easy-to-use device, targeting the most common applications. With the rise of [GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units), GPU also becomes a popular choice for deep learning and high-performance computing, but the downside of GPU is its large power consumption. On the other hand, [field-programmable gate array (FPGA)](https://en.wikipedia.org/wiki/Field_programmable_gate_array) is more energy-efficient than CPU and GPU, but requires lots of programming efforts to achieve high performance.

GA144 can be classfied as a *manycore* processor, which can be placed between GPU and FPGA. Another spatial architecture called [coarse-grained reconfigurable array (CGRA)](https://cccp.eecs.umich.edu/research/cgra.php) also has lower power consumption than CPU and GPU, but is hard to program.

In the rightmost side of this figure is [application-specific integrated circuit (ASIC)](https://en.wikipedia.org/wiki/Application-specific_integrated_circuit). It provides really high performance and low energy consumption, but is designed for a specific application and cannot be reconfigured for other applications. Google's [TPU](https://cloud.google.com/tpu) and Huawei's [Ascend NPU](https://www.huawei.com/en/news/2019/8/huawei-ascend-910-most-powerful-ai-processor) are both examples of ASIC.

A consensus is that there does not exist an one-size-fits all processor. Different devices may have their own strengths and weaknesses. Combining different technologies in a large chip seems to be a trend of future devices. Apple's M1 chip has a 16-core [Neural Engine](https://www.apple.com/newsroom/2020/11/apple-unleashes-m1/) which is also a type of NPU. Nvidia's [Turing GPU](https://www.nvidia.com/en-us/data-center/tensor-cores/) has a specialized Tensor Core architecture which greatly accelerates low-bitwidth matrix multiplication but requires special program snippets to run on that.

![](devices.png)

We also discuss in class why GA144 is not widely adopted nowadays and what makes a device become popular. The first thing should be programmers' productivity. If a device is easy to use and debug, even its performance is not that good, it can still be accepted by many people. The second thing is the cost of the device. Based on the [price](http://www.greenarraychips.com/home/products/index.php) listed on the GreenArray website, it costs $20 per chip, which is a very reasonable price and can be massively manufactured, so the main problem of low acceptance is probably programmability.


## Motivation
Based on the background we discussed above, the authors propose the following challenges that motivate them to develop a compiler for GA144:
1. <u>Spatial architecture is hard to program.</u> The data placement, communication, and computation all need to be manually specified by the programmers using very low-level language.
2. <u>Classical compiler may not be able to bridge the abstraction gap of low-power computing.</u> For one thing, designing compilers for new hardware is hard since no well-known optimizations can be applied. For the other, the GA144 architecture was still evolving at that time, so the compiler should also evolve fast to keep up with the pace.
3. <u>Program synthesis is hard to scale to large programs.</u>

Therefore, the authors propose a synthesis-aided compiler Chlorophyll to solve the above challenges. 

## Methods

In this section, I will talk about the four stages of the compiler.

### Program Partition

is code separation a synthesis problem?

### Layout and Routing

### Code Separation

### Code Generation


## Further Discussion


## Programming Model


## Synthesis and Compilation


## Reference
[^1]: James Bornholt, Emina Torlak, [Scaling Program Synthesis by Exploiting Existing Code](https://www.cs.utexas.edu/~bornholt/papers/scalesynth-ml4pl15.slides.pdf)