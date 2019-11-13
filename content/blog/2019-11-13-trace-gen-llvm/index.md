+++
title = "Generating traces from llvm"
extra.author = "Philip Bedoukian and Sachille Atapattu"
+++

Let's generate traces from llvm!
---------------------------

## Overview

In this project we use llvm to generate traces and execute them. We use llvm passes to log all the traces in a program and use another pass to create an executable that only executes a selected trace. This could be useful to implement efficient trace selection, or to implement trace-based llvm passes in general.

## Trace detection

To determine all the traces in a given program, we first use llvm functions to detect branch instructions. We isolate branch instructions that diverge control flow and then insert a function call to a runtime library to log the condition upon divergence. At runtime, when the conditions to the diverging branch instructions are resolved, the sequence of branch selection at execution is generated as a csv file of boolean values. This sequence of branch selection forms a trace through the entire program taken at the execution of trace detection runtime.

This llvm pass was created broadly following the "Linking With a Runtime Library" section from [LLVM for Grad Students](https://www.cs.cornell.edu/~asampson/blog/llvm.html) blogpost. However, a `Value` was created from `FunctionCallee` object before passing into `CreateCall` function from [`IRBuilder`](https://llvm.org/doxygen/classllvm_1_1IRBuilder.html) to avoid a casting error. This pass is registered as `-skull` in `Skull.cpp`.

The next step in trace detection was detecting backedges and clippling off isolated traces from the full execution trace. LLVM provides great tools to achieve this through "interactions between passes" detailed in [Writing an LLVM Pass](http://llvm.org/docs/WritingAnLLVMPass.html#specifying-interactions-between-passes). Using [LoopPass](https://llvm.org/doxygen/classllvm_1_1Loop.html) analysis to gather loop information, LLVM provides a nice way to detect loops and whether each basic block is within a loop or not. However, accessing entry point to the loop and exit point turned out to be hard. The pass is expected to mark the start and end points of each loop in the trace log, thereby enabling trace selection and optimizing. Work in progress on this pass can be found using `-pelvis` in `Pelvis.cpp`.

## Trace selection

## Trace generation

## Evaluation
