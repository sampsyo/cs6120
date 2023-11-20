+++
title = "Chlorophyll: synthesis-aided compiler for low-power spatial architectures"
[extra]
bio = """
  Matthew Hofmann is a 2nd year Ph.D. student researching design automation for reconfigurable hardware. Yixiao Du ... TODO(yxd).
"""
[[extra.authors]]
name = "Matthew Hofmann"
[[extra.authors]]
name = "Yixiao Du"
+++

Chlorophyll is a synthesis-aided compiler that targets a spatial array of ultra-low power scalar cores. Chlorophyll takes a subset of C as input and maps the computation to the hardware in a process not unlike FPGA high-level synthesis. Each core of the GreenArrays GA144 is extremely barebones. They have their own proprietary ISA, it only has 64 18b words of memory, and the cores are not clocked synchronously. While the paper's compilation problem is posed around an extremely niche hardware architecture, the authors' contributions of modular superoptimization is deserving of a close look.

In this blog post, we will breakdown the subproblems of program synthesis and qualitatively judge how effective their synthesizer is at optimizing programs for spatial architectures. To summarize, the main two claims the authors make are (1) that the loss of QoR (quality of results) is worth the increased level of automation and (2) that modular superoptimization is effective enough at finding peephole optimizations like strength reduction and bit-hacks.

Full citation:
```
Phitchaya Mangpo Phothilimthana, Tikhon Jelvis, Rohin Shah, Nishant Totla, Sarah Chasins, and Rastislav Bodik. 2014. Chlorophyll: synthesis-aided compiler for low-power spatial architectures. SIGPLAN Not. 49, 6 (June 2014), 396–407. https://doi.org/10.1145/2666356.2594339
```

## Background

TODO

## The Architecture

TODO

## The Compiler

* Design Partitioning
* Placement
* Code Separation
* Code Generation

## Modular Superoptimization

TODO

## Discussion

TODO
