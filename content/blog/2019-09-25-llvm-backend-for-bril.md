+++
title = "LLVM JIT Compiler for Bril"
extra.author = "Shaojie Xiang & Yi-Hsiang Lai & Yuan Zhou"
extra.bio = """
  [Shaojie Xiang](https://github.com/Hecmay) is a 2nd year ECE PhD student researching on programming language and distributed system. 
  [Yi-Hsiang (Sean) Lai](https://github.com/seanlatias) is a 4th year PhD student in Cornell's Computer System Lab. His area of interests includes electronic design automation (EDA), asynchronous system design and analysis, high-level synthesis (HLS), domain-specific languages (DSL), and machine learning. 
  [Yuan Zhou](https://github.com/zhouyuan1119) is a 5th year PhD student in ECE department Computer System Lab. He is interested in design automation for heterogeneous compute platforms, with focus on high-level synthesis techniques for FPGAs.  
"""
+++

Bril is a concise intermediate representation language, which is powerful enough to describe most common arithmetic operations (e.g., add, mul, div, and other control flow instructions). In this project, we aim to extend the reachability of Bril IR to different backend devices by compiling Bril programs to LLVM IR. We execute the generated LLVM IR via LLVM execution engine to verify its functional correctness. Finally, we compare the runtime between LLVM JIT compilation and Bril interpreter.

### Methodology 

To compile a Bril program into LLVM IR, we first take the program in JSON format and have it analyzed by our compiler. The overall workflow is similar to what we do for data flow analysis in class. One thing to notice here is that, during the class, we have not mentioned static single assignment (SSA), wihch is an IR property requiring each variable to be assigned exactly once. Multiple assignments to same variable create new versions for that variable. SSA is essential when we have multiple assignments to a single variable. Namely, we need to create phi nodes in cases where we have branches. However, Bril is not an SSA-form IR where multiple assignments overwrite the variable without creating new identifiers. To compile the Bril IR into SSA-form LLVM IR, we make each assignment a unique memory store. Similarly, each variable read becomes a memory load.

1. Create a basic block mapping: Given the Bril IR in JSON representation, we create empty LLVM basic blocks according to block labels. Meanwhile, we maintain a mapping between label strings and LLVM basic block pointers. We also create a flag to mark whether a basic block is used or not.
2. Insert instructions into blocks: We traverse the empty basic blocks and insert instructions into them. Each basic block should end with a valid terminator (i.e., jmp, br, or ret). The insertion process will terminate after encountering the first terminator. All following instructions under the same label will be ignored since this code is dead and will not be executed in any condition. 

```markdown
label:
  br cond b1 b2
  # inst n ignored
  n: int = mul a b;  
b1:
  m: int = const 5;
  print v;
b2:
  jmp end;    
```

3. Dump LLVM code and run through JIT compilation: We allow the users to dump the generated LLVM IR for easy inspection. After that, we compile the code with LLVM execution engine and verify the outputs by comparing the results produced by Bril interpreter.

### Implementation details

 In this section, we briefly describe the implementation details for each step in LLVM code generation process. To get a global idea of how different components are linked and executed in the program, we need a data structure recording basic blocks and the actual values each instruction takes. Here is a list of the data structure we create. Basically, we categorize them into two classes: one is for block level and the other for instruction level.

```c++
using BasicBlockFlag_T = pair<llvm::BasicBlock, bool>;
using BasicBlockMap_T = map<string, BasicBlockFlag_T>;
using VarToVal_T = map<string, llvm::Value>;
```

The `BasicBlockFlag_T` is used to track whether a block has been visited or not. We will remove the unused basic blocks when traversing the whole program. Following is a simple example of removing redundant basic blocks.

```markdown
cond bool: lt a b;
br cond b1 b3
b1:
  jmp end;
b2: 
  jmp end;
b3:
  jmp end;
```

The `b2` branch will not be reached in any condition. During the instruction insertion process, the block labeled `b2` will not be marked and thus we know that it is redundant. This basic optimization helps reduce the executable size.

The `BasicBlockMap_T` structure constructs a mapping from block name to actual LLVM blocks. When the instruction visitor traverses the control data flow graph, we create a new LLVM basic block every time we find a new label (except for the special case where the entry basic block has no label). The unordered mapping from label name to LLVM basic blocks will be created and saved for later use.

The `VarToVal_T` structure tracks the pointer of each destination variable.

```c++
// create a alloca llvm value using IRBuilder
llvm::Value* val = builder->CreateAlloca(t_int_, llvm::ConstantInt::getSigned(t_int_, 1));
// save the allocated value into the VarToVal map
(*varToVal)[destination] = val;
```

Special note for the print instruction in Bril: we create an LLVM function call with integer return data type, and pass in `%d` and the actual LLVM value to be printed as arguments. Then we build a `CreateCall` node with LLVM IRBuilder so that the print function can be realized in LLVM program.

### Experiment Results

Our program is in [one of Bril's forks](https://github.com/seanlatias/bril/tree/master/codegen-llvm), under the ``codegen-llvm`` folder. To compile our JIT compiler, ``make`` is the only command needed. Our program takes in two variables. One is the input Bril program in JSON format and the other is the output LLVM file (usually ends with `.ll`).

To verify the correctness of the generated LLVM IR, we develop several test cases, which cover most commonly used arithmetic and control flow instructions, as well as some corner cases where the program has some redundant instructions that could be removed. The test example is shown as followed (it can also be found under the `codegen-llvm` folder):

```markdown
main {
  a: int = const 42;
  b: int = const 22;
  v: int = add a b;
  m: int = mul v b;
  cond: bool = lt a m;
  br cond b1 b2;
  # inst n removed automatically
  n: int = mul a b; 
b1:
  m: int = const 5;
  print v;
  # br to b2 inserted here
b2:
  jmp end;
end:
  # print func in llvm
  print a;
}
```

We wrote the program in text representation for Bril and get the canonical JSON form Bril program with `bril2json`. By running the following commands, the JSON file will be generated and analyzed. Our compiler then generates the LLVM code and print out the LLVM program into the destination file.

```shell
cat test.bril | bril2json > test.json
./bril-llvm test.json test.llvm
```

By observing the generated LLVM code, we can see that at the very end of branch `b1`, a new instruction is added to avoid the issue where the basic block is missing a terminator. Moreover, the print function of Bril is transformed into an LLVM function call with corresponding variables passed in as arguments.

```llvm
  b1:                                               ; preds = %0
    %13 = alloca i64, i64 1
    store i64 5, i64* %13
    %14 = load i64, i64* %5
    %15 = call i64 @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @0, i32 0, i32 0), i64 %14)
    br label %b2
  b2:                                               ; preds = %b1, %0
    br label %end
  end:                                              ; preds = %b2
    %16 = load i64, i64* %1
    %17 = call i64 @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @1, i32 0, i32 0), i64 %16)
    ret i32 0
  }
```

After verifying the correctness of the code generator, we also compare the performance of LLVM simulation and the Bril interpreter. The performance is measured with profiling tool in Linux and C++. We run the same program for 10 times and take the average runtime. For the test program with a regular for loop iteratively computing one multiply operation for 1 billion times, the LLVM interpreter runs about 10 times faster than the Bril interpreter. The average runtime is 0.47 seconds and 0.05 seconds for Bril and LLVM interpreter respectively. The LLVM execution engine achieves approximately 1000x speedup over the Bril interpreter without optimizing the loop inside. We can expect higher speedup if the loop is optimized away.
