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
```
# A kernel that accumulates values across invocations
def stateful_kernel(x: int32) -> int32:
    # 'acc' retains its value between calls
    acc: stateful(int32) = 0
    acc = acc + x
    return acc
```
```
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

```
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

# Hardest Parts and Future Work

# Additional Evaluation

What was the goal?
What did you do? (Include both the design and the implementation.)
What were the hardest parts to get right?
Were you successful? (Report rigorously on your empirical evaluation.)


my [fork](https://github.com/sunwookim028/allo)
Here's a [video](https://youtu.be/2dKNX0L-iG8?si=Jlyv5rDRoa-c0X9h)
