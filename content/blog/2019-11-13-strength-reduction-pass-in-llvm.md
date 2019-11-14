+++
title = "Strength Reduction Pass in LLVM"
extra.author = "Shaojie Xiang & Yi-Hsiang Lai & Yuan Zhou"
extra.bio = """
  [Shaojie Xiang](https://github.com/Hecmay) is a 2nd year ECE PhD student researching on programming language and distributed system. 
  [Yi-Hsiang (Sean) Lai](https://github.com/seanlatias) is a 4th year PhD student in Cornell's Computer System Lab. His area of interests includes electronic design automation (EDA), asynchronous system design and analysis, high-level synthesis (HLS), domain-specific languages (DSL), and machine learning. 
  [Yuan Zhou](https://github.com/zhouyuan1119) is a 5th year PhD student in ECE department Computer System Lab. He is interested in design automation for heterogeneous compute platforms, with focus on high-level synthesis techniques for FPGAs.  
"""

+++

Strength reduction is basically an approach to substitute expensive operations (e.g. multiplication, redundant updates for induction variables) to computationally cheaper ones (e.g. shifting, reused variable). For this course project, we implemented the strength reduction technique for loops as an LLVM pass. The algorithm will identify induction variables and reduce the expensive computation to save the runtime. The source code can be found [here](https://github.com/Hecmay/llvm-pass-skeleton)

### Methodology 

#### Preprocessing 

In modern compiler frameworks, strength reduction usually follow a series of preprocessing passes. For example, LLVM has a loop canonicalization and simplification pass, which makes sure that the loop has one and only one induction variable, canonicalizes the loop induction variable to always start from zero and have unit stride, and creates pre-headers for loops that have multiple entrances. 

```c++
// before simplification 
int i = 0;
while( i < 10 ) {
  int j = 3 * i + 2;
  a[j] = a[j] - 2;
  i = i + 2;
}

// after simplification
int i = 0;
while( i < 5 ) {
  int j = 6 * i + 2;
  a[j] = a[j] - 2;
  i = i + 1;
}
```

Apart from loop simplification we also apply dead code elimination and induction variable simplification passes, canonicalizing the loops into the standard form, which makes it easier to analyze and optimize. 

#### Loop and Induction Variable Analysis

The first step is to analyze loops in the program. A loop is composed of basic blocks (typically can be categorized as loop condition). Here is an example for optimizing the loop with strength reduction: the original program updates the array index j with a multiply operation every iteration. 

```c++
int i = 0;
while( i < 10 ) {
  int j = 3 * i + 2;
  a[j] = a[j] - 2;
  i = i + 2;
}
```

To optimize the loop, we can update the j with a pre-computed stride, which will substitute the original expensive multiply operations with cheaper add operation, as shown in the following snippet.

```c++
int i = 0;
int j = 2; // j = 3 * 0 + 2
while( i < 10 ) {
  j = j + 6; // j = j + 3 * 2
  a[j] = a[j] - 2;
  i = i + 2;
}
```

To locate the variables and perform the substitution, we need to create a map describing how each induction variable is dependent on the canonical induction variable. Say variable k = 3 * i + 1, then we denote k as a triple <i, 3, 1> and create a map mapping the variable to the corresponding triple:

Map { Value => (base, multiplicative factor, additive factor) }

We scan loop body to find all basic induction variables, which are identified with followed two conditions : 

* (1) if finding an assignment of form k = b * j where j is an induction variable with triple <i, c, d>, then add {k : <i, b* c, d>} to the mapping 
* (2) if finding an assignment of form k = j + b where j is an induction variable with triple <i, c, d>, then add {k : <i, c, d + b>} to the mapping 

The algorithm will stop until the induction variable set size does not increase any more. Then we can create new phi-nodes in the loop and have it updated by adding a pre-computed stride. In the realistic applications loops are nested, namely there can be different levels of sub-loops existing in the program. The strength reduction algorithm should be robust enough to extract the induction variables in all sub-loops and substitute them with less expensive operations.

### Implementation Details 

#### Preprocessing 

For the loop preprocessing, we create a function pass manger to include all necessary passes we want to apply. The pass manager is instantiated with the module encapsulating this LLVM function. 

```c++
legacy::FunctionPassManager FPM(module);
FPM.add(createConstantPropagationPass());
FPM.add(createDeadCodeEliminationPass());
FPM.doInitialization();
bool changed = FPM.run(F);
FPM.doFinalization();
```

The post processing loops are canonical in terms of stride and loop count initial value. With the canonicalized loops we do the phi-node analysis and create a new phi-node for non-canonical induction variables. These variables will be updated in a less expensive way while keeping same functionality.

#### Induction Variable Analysis
This section details how we do the loop analysis and extract useful information for mitigating the computationally expensive operations in loops.  In the `getAnalysisUsage` function in the function pass struct, we added the required pass  wrapper and generate a `LoopInfo` object, which contains all loop related information of the LLVM Function we are processing.

```c++
void getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
} 
```
For each loop in the program, we create a map to record all induction variables. Each induction variable (as an LLVM Value) is mapped to a factorized representation of other inductions, as mentioned the first section. For example, the variable equaling to 3*i + 1 will be stored as <i, 3, 1>, where i is the loop count.  

```c++
map<Value*, tuple<Value*, int, int> > IndVarMap;
// collect all basic indvars by visiting all phi nodes
for (auto &B : F) 
  for (auto &I : B) 
    if (PHINode *PN = dyn_cast<PHINode>(&I)) 
      IndVarMap[&I] = make_tuple(&I, 1, 0);
```

We can collect all induction variables by iterating the loops, adding new induction variables to the map if the following conditions are satisfied: (1): if the value is computed with an Add / Sub instruction on a constant and a induction variable that is already in the map (2): if the value is computed with an Mul instruction on a constant and a induction variable that is already in the map. The algorithm will iterate the loops until the map size does not increase any more. 

#### Strength Reduction 

With the induction variable mapping information available, we can analyze which induction variable can be replaced in the loop. Namely if there is a non-canonical induction variable in the loop (e.g. k = 3 * i + 8 where index i is another induction variable), we can create a new update rule for it.

To track all the induction variables to replace with, we create a map from LLVM Values (that are remaining to be replaced) to corresponding phi-nodes. These PHINodes are created in the loop header right after the canonical induction variable. Their initial values are calculated based on the canonical induction variable e.g. j(0) = 3 * i(0) + 1 = 1 as the when the loop is executed in the first time. And for the other incoming value, we compute the stride based on the mapping information from  `IndVarMap`. For example, a induction variable k will be updated with k = k + 6, if k is mapped to triple <i, 3, 1> and i has a update stride of 2 (i.e. every time i increased by 2, k will be increased by 6 as a result).

```c++
// replace all the original uses with phi-node
for (auto &B : F) 
  for (auto &I : B) 
    if (PhiMap.count(&I)) 
      I.replaceAllUsesWith(PhiMap[&I]);
```

After creating all the phi-nodes for induction variable substitution, we will update all the original values with the newly created PHINodes to complete the optimization. 

### Experiment Results

We use realistic benchmark [embench-iot](https://github.com/embench/embench-iot), which is a benchmark suite designed to test the performance of deeply embedded systems. The strength reduction pass is performed on each program to evaluate its correctness and efficiency. Experiments are performed on a server with an 2.20GHz Intel Xeon processor and 128GB memory. All programs are single-thread.

To run the optimized program, we first use clang to emit LLVM IR of original program with all optimization pass disabled. Then the IR is passed into LLVM opt, optimized with our pass and compiled into bitcode and objects. Finally we compile the object files into binary and run on physical machines.

```shell
# generate LLVM IR for original program 
clang -c -emit-llvm -O0 -Xclang -disable-O0-optnone benchmark.c $EMBENCH_DIR/support/*.c -I/path/to/benchmark \
-I$EMBENCH_DIR/support -DCPU_MHZ=1

# apply pass with llvm opt and compile to bitcode
opt -S -load build/skeleton/libSkeletonPass.so -mem2reg -sr opt.ll -o opt.bc
# compile to obj & binary 
llc -filetype=obj opt.bc;
gcc *.o -lm; 
./a.out
```

The LLVM pass is first compiled into a shared library, which is loaded into LLVM opt as a compiler pass (with -sr argument, which is custom name for our pass). We first run the pass with a toy example of vector addition, and verified its correctness by comparing the IR before and after applying the pass. We also did comprehensive experiments for each design in the benchmarks. Here is the experiment results:

| Benchmark | Original (s) | Optimized (s) | Sppedup |
|:---------:|:------------:|:------------:|-----------|
|aha-mont64|0.261|0.244| 1.070 |
|crc32|0.598|0.598| 1 |
|cubic|0.018|0.018| 1 |
|edn|0.439|0.389| 1.129 |
|huffbench|0.564|0.027| 20.889 |
|matmult-int|0.628|0.579| 1.085 |
|minver|0.067|0.078| 0.859 |
|nbody|0.01|0.015| 0.667 |
|nettle-aes|0.316|wrong| N/A |
|nettle-sha256|0.348|0.364| 0.956 |
|nsichneu|0.312|0.314| 0.994 |
|picojpeg|0.545|stuck| N/A |
|qrduino|1.183|1.136| 1.041 |
|sglib-combined|0.612|wrong| N/A |
|slre|0.729|wrong| N/A |
|st|0.064|0.049| 1.306 |
|statemate|0.372|0.401| 0.928 |
|ud|0.55|0.538| 1.022 |
|wikisort|0.215|0.276| 0.779 |
||||  |

Except for few designs that our pass cannot handle, the average speedup is 1.211x over the original baseline program. For few cases, our pass cannot get the correct result or may generate program running into a deadlock. The reasoning behind that might be that 
