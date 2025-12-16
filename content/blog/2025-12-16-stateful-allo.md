+++
title = "Modeling Stateful Accelerators in Allo"
[extra]
bio = """
  Sunwoo Kim is interested in AI hardware. He is pursuing ECE PhD at Cornell.
"""
[[extra.authors]]
name = "Sunwoo Kim"
link = "https://sunwookim028.github.io/"
+++

# Context and Vision
Computer architects are building accelerator chips for demanding domains like AI. 
[Allo](https://cornell-zhang.github.io/allo/) is a Python-embedded domain-specific language to facilitate the design of such accelerators in a modular and composable manner, with users across industry and academia.
However, the current language doesn't have a support for stateful variables, hence users cannot model practical accelerators that holds intermediate data or results on chip.
We aimed to add a language feature for users to specify stateful variables with compilation flow to lower them to static variables in Vitis HLS backend.

# Summary of Results
We successfully added a new type qualifier in Allo to specify stateful variables and compilation flow to lower them into static variables in Vitis HLS, bridged by MLIR.
We demonstrated programmable stateful accelerator designs for scalar and tensor arithmetics.
Our most up-to-date code is in this [fork](https://github.com/sunwookim028/allo) and is being merged to the upstream.
Here's a presentation [video](https://youtu.be/2dKNX0L-iG8?si=Jlyv5rDRoa-c0X9h) for a different course (ECE6775) where this project was also presented.

# Acknowledgement
This work was done with Joseph Maheshe, Jifeng Wu, Jenny Lee from ECE6775 project team.
My specific contributions were in suggesting the frontend syntax, implementing the MLIR to HLS backend and writing and running stateful scalar and tensor arithmetic accelerators.

# Design and Implementation
<img src="./2025-12-16-stateful-allo/overview.png" alt="allo to mlir to hls" width="310"/>
This is the overview of our compilation flow.
We'll use this scalar accumulator example to explain each step.

```python
# A kernel that accumulates values across invocations
def stateful_kernel(x: int32) -> int32:
    # 'acc' retains its value between calls
    acc: stateful(int32) = 0
    acc = acc + x
    return acc
```
In the frontend, adding the `stateful` qualifier specifies that the variable is a stateful one.

```mlir
module {
  // Global storage (persistent)
  memref.global "private" @__stateful_stateful_kernel_acc_1 : memref<i32> = dense<0>
  func.func @stateful_kernel(%arg0: i32) -> i32 {
    // 2. Local handle to storage
    %0 = memref.get_global @__stateful_stateful_kernel_acc_1 : memref<i32>
    // 3. Load/store operations using the handle
    %1 = affine.load %0[] {from = "acc"} : memref<i32>
    // ... arithmetic operations ...
    affine.store %5, %0[] {to = "acc"} : memref<i32>
    return %6 : i32
  }
}
```
They are then translated into global variables in IR, marked with a naming pattern `_stateful`.

```c++
void stateful_kernel(
  int32_t v0,
  int32_t *v1
) {    // L3
  static int32_t __stateful_stateful_kernel_acc_1 = {0};    // L2
  // placeholder for const int32_t __stateful_stateful_kernel_acc_1    // L4
  int32_t v3 = __stateful_stateful_kernel_acc_1;    // L5
  ap_int<33> v4 = v3;    // L6
  ap_int<33> v5 = v0;    // L7
  ap_int<33> v6 = v4 + v5;    // L8
  int32_t v7 = v6;    // L9
  __stateful_stateful_kernel_acc_1 = v7;    // L10
  *v1 = __stateful_stateful_kernel_acc_1;    // L11
}
```
Then they are translated into static variables in the backend.
HLS codegen emits static keyword for such marked stateful variables.

# Additional Evaluation
By measuring execution time, we discovered that the overhead of copying the state in host adds ~20% overhead.

Also, we demonstrated a programmable accelerator design here.

```python
import allo
from allo.ir.types import int32, stateful, uint8

MEM_SIZE = 4
OP_H2D = 0    # memcpy from host to accelerator (device)
OP_D2H = 1    # memcpy from accelerator to accelerator (device)
OP_ADD = 2    # compute addition on-chip
OP_MUL = 3    # compute multiplication on-chip

def int32_add(op1: int32, op2: int32) -> int32:
    return op1 + op2

def int32_mul(op1: int32, op2: int32) -> int32:
    return op1 * op2

def arith_processor(op: uint8, inval: int32, addr: uint8) -> int32:
    mem: stateful(int32[MEM_SIZE]) = 0
    retval: int32
    if op == OP_H2D:
        mem[addr] = inval
        retval = 99 # random value
    if op == OP_D2H:
        retval = mem[addr]
    if op == OP_ADD:
        mem[addr] = int32_add(mem[addr], mem[addr + 1])
        retval = mem[addr]
    if op == OP_MUL:
        mem[addr] = int32_mul(mem[addr], mem[addr + 1])
        retval = mem[addr]
    return retval
```
