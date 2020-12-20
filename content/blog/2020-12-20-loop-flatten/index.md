+++
title = "A Loop Flattening Pass in LLVM"
[extra]
bio = """
  [Hanchen Jin](https://hj424.github.io/) is a 4th year Ph.D. student in the Computer Systems Lab at Cornell. His research interests are compute architecture, hardware-software co-design, parallel computing with dedicated accelerators, and High-Level Synthesis (HLS).
"""
latex = true
[[extra.authors]]
name = "Hanchen Jin"
+++

## Background
1. LLVM Compiler Infrastructure<br/>
This open-source project was started by the researchers from UIUC in 2000. It is a collection of the modular and reusable compiler. The goal of this project is to provide a modern, SSA-based compilation strategy capable of supporting both static and dynamic compilation of arbitrary programming languages. With the hard work from a diverse community, the LLVM compiler has become a powerful and stable tool that is widely applied in many commercial and academic research projects. LLVM is a robust compiler with abundant APIs which is easy for us to hack it and develop a more efficient compiler for our own requirements. Thus, we select the LLVM compiler as the framework for this project. If you are interested in the LLVM, please check out the links in the References section.

2. Loop Optimizations<br/>
As a programmer, we always want to finish running our programs as fast as possible. However, with the end of Moore’s law around 2000, the computing power for a single CPU is limited by its physical features. To further improve the performance, the multi-core system with parallel computing has become a popular solution. But it requires the programmer to manually optimize the code to achieve higher performance. For some applications, we really want the compiler itself can figure out the heated piece of the code and automatically optimize it.

As we know, loops are usually the heavy part during execution, which consumes most of the runtime. Thus, many strategies have been applied to speed up the execution time of the loops. For example, Loop Invariant Code Motion (LICM) removes the loop invariant instructions out of the loop to avoid perform useless instructions. For more details about all available loop optimizations, please check out [this link](https://llvm.org/docs/Passes.html). And for the explanation of loop optimizing terminologies, please check out [this link](https://llvm.org/docs/TransformMetadata.html).

## Motivation
As we mentioned before, with the multi-core compute architecture, we would like to leverage the parallelism of the code to get better performance. As for some loops, there is no data dependency around multiple iterations, which is a good fit for loop unrolling optimization. Specifically, multiple loop iterations can be executed independently at the same time. Therefore, in this section, we will introduce the loop unrolling optimizations and introduce the motivation of adding the loop flattening optimization.

1. Loop unrolling<br/>
In LLVM, this optimization unrolls the loop with the given unroll parameters. To be more specific, it will transform the following piece of code:

```javascript
for (int i = 0; i < n; i++) { // original loop
  Stmt(i);
}
```

Into this piece of code:

```javascript
for (int i = 0; i < n; i+=4) { // unrolledloop
  Stmt(i);
  Stmt(i+1);
  Stmt(i+2);
  Stmt(i+3);
}

for (int i = 0; i < m; i++) { // remainder loop
  Stmt(i);                            // m = n % 4 
}
```

2. Loop unroll-and-jam<br/>
In LLVM, this optimization unrolls the nested loop with the given unroll parameters. To be more specific, it will transform the following piece of code:

```javascript
for (int i = 0; i < n; i++) { // original outer loop
  Outer_Stmt(i);
  for (int j = 0; j < m; j++) { // original inner loop
Inner_Stmt(i);
  }
}
```
Into this piece of code:

```javascript
for (int i = 0; i < n; i+= 2) { // unrolled outer loop
  Outer_Stmt(i);
  Outer_Stmt(i+1);
  for (int j = 0; j < m; j+= 2) { // unrolled inner loop
Inner_Stmt(i);
Inner_Stmt(i+1);
  }
}
```

3. Loop flattening<br/>
As illustrated by its name, this optimization will flatten the nested loop into a single-level loop. 
To be more specific, it will transform the following piece of code:
```javascript
  for (int i = 0; i < n; i++) { // unflattened nested loop
    for (int j = 0; j < m; j++) {
      Stmt(f(i, j))
  }
}
```
Into this piece of code:

```javascript
for (int i = 0; i < n*m; i+= 2) { // flattened loop
  Stmt(f(i))
}
```

After flattening the loop, the loop bound becomes larger so that we can apply larger unrolling parameters to further explore the parallelism. Also, this technique is necessary for some optimizations in High-Level Synthesis tools. For example, suppose we have a nested for loop with variable loop bound, the tool cannot automatically apply the pipeline optimization because it has no knowledge about how to deal with the inner loop. Thus we need to manually flatten the nested loop and then apply the pipeline pragma. To learn more about the importance of the loop flatten technique for pipelining, please check out [this link](https://people.ece.uw.edu/hauck/publications/LoopFlattening.pdf).


## Implementation
The idea is quite straightforward but the implementation is not that easy. To implement this pass, we should perform the following modifications.

1. Update the loop bound and increment <br/>
As the first step, we should update the loop bound for the outer loop. Suppose the loop bound for the inner loop and outer loop are m and n with the same increment, we can change the loop bound for the outer loop to be `m*n` and keep the same increment.

2. Extract the statements inside the inner loop to the outer loop<br/>
In this step, we extract all the statements inside the inner loop and put them into the outer loop. At the same time, we should detect the usage of the inner loop variable and then remove it with the correct one after flattening the loop. For example, suppose we have a piece of code like this inside the inner loop: `i*n+j`, whereas `i` and `j` are the loop variables, `n` is the outer loop bound, and the increment for both inner loop and outer loop is 1. To correctly flatten this loop, we should replace the statement `i*n+j` with `i`. 

3. Delete the inner loop<br/>
After replacing the statement from the inner loop to the outer loop, we can safely remove the inner loop by deleting the corresponding basic blocks.

Finally, we should check the equivalence of the code, i.e., the flattened loop should perform the same execution as the original nested loop. This can be verified by testing the pass with benchmarks. 

During the implementation, I got stuck in the second step, i.e., extract the statements inside the inner loop to the outer loop. The generated code performs weird when extracts the loop variant expressions from the inner loop to the outer loop. 

Thus, to simplify the problem, I decide to change the inner loop bound instead of the outer loop bound. This is quite easy because we only need the first step for “flattening” the loop. But for the pass I build, it only works for the “perfect” nested loop, i.e., the nest loop that only contains statements inside the inner loop. 
```javascript
for (int i = 0; i < n; i++) { // unflattened “perfect” nested loop
    // no statement here
    for (int j = 0; j < m; j++) {
      Stmt(f(i, j))
```

My naïve pass is available in [this link](https://github.com/hj424/bril/tree/master/tasks/project/llvm-pass-skeleton).

## Evaluation
1. Correctness <br/>
As for the correctness of the pass, we should focus on two points: the functionality of the optimized code and the correctness of the unrolled loop. These two points are being checked by running the simple benchmarks with the clang profiling tools. For more details, please check out [this link](https://clang.llvm.org/docs/SourceBasedCodeCoverage.html).

 As documented in the readme file of the code repository, you can run the code with my bash script.

To elaborate the evaluation, let me use the following source code as an example: 

```javascript
#include <stdio.h>
#include <stdlib.h>

#define N 11
#define M 20

int main() {
  int init_val = 30;
  int res[M*N];
  LOOP:
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      res[i*M+j] = init_val;
    }
  }
  printf("Multiplication res: %d; \n", res[0]);
  return 0;
}
```

After optimizing this loop with my pass, the runtime profiling is showed here.

```javascript
Multiplication res: 30; 
    1|       |#include <stdio.h>
    2|       |#include <stdlib.h>
    3|       |
    4|      2|#define N 11
    5|    441|#define M 20
    6|       |
    7|      1|int main() {
    8|      1|  int init_val = 30;
    9|      1|  int res[M*N];
   10|      1|  LOOP:
   11|      2|  for (int i = 0; i < N; i++) {
   12|    221|    for (int j = 0; j < M; j++) {
   13|    220|      res[i*M+j] = init_val;
   14|    220|    }
   15|      1|  }
   16|      1|  printf("Multiplication res: %d; \n", res[0]);
   17|      1|  return 0;
   18|      1|}
```
The first line is the sample result I printed out to verify the correctness of the optimized code. And then the following code is the optimized one with the updated inner loop bound. As you can see from line 11, the outer loop was being executed only 1 time. The number is 2 here means the second time, it checks the loop bound and exits the outer loop. And accordingly, the inner loop is being executed `N*M` times, which is 220 from line 13. This naïve pass is being tested with other small benchmarks I created with perfect nested loop and passed all the tests.

2. Performance gain<br/>
As we mentioned before, the loop flattening pass itself does not offer any benefits for performance. But this technique helps to better explore the parallelism when combining with the loop unrolling technique. Specifically, we can apply larger unroll parameters to further explore the parallelism across the loop iterations. Also, this technique is used to support other optimizing pragmas (like loop pipelining in HLS tools) for better performance.


## Conclusion
In summary, loop flattening is a useful optimizing technique for supporting loop unrolling and other HLS related optimizations. LLVM is an awesome infrastructure that provides us powerful APIs for creating our own pass. Finally, I would like to mention that the loop flattening pass has just been merged into the official LLVM github repository (LLVM v12.0.0). Here is the [source code](https://llvm.org/doxygen/LoopFlatten_8cpp_source.html). Unlike my naive implementation, this pass can actually flatten the loop as shown in the following example:

```javascript
 // from nested loop:
 for (int i = 0; i < N; ++i)
   for (int j = 0; j < M; ++j)
     f(A[i*M+j]);
 // into one loop:
 for (int i = 0; i < (N*M); ++i)
   f(A[i]);
```

However, it also has some constraints which are elaborated in the [source code](https://llvm.org/doxygen/LoopFlatten_8cpp_source.html).

## References
[1] The LLVM Compile Infrastructure: https://llvm.org/ <br/>
[2] LLVM user reference manual: https://llvm.org/docs/LangRef.html<br/>
[3] LLVM developer reference manual: http://llvm.org/doxygen/<br/>
[4] LLVM official github repository: https://github.com/llvm/llvm-project<br/>
[5] Loop flatten techniques applied in software pipelining (converting loop level parallelism into instruction parallelism): https://people.ece.uw.edu/hauck/publications/LoopFlattening.pdf<br/>


