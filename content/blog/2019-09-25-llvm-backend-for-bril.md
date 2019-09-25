
+++
title = "Project Report 1: LLVM IR transformer"
extra.bio = """

"""
[[extra.authors]]
name = "Shaojie Xiang"
[[extra.authors]]
name = "Yi-Hsiang (Sean) Lai"
[[extra.authors]]
name = "Yuan Zhou"
+++

## Project Report 1: LLVM IR transformer 

Bril is a concise intermediate representation language, which is powerful enough to describe most common arithmetic operations (e.g. add, mul, dev and other control flow instructions). In this project, we aim to extend reachability of Bril IR to different backend devices by converting it into LLVM IRs. The generated LLVM IR program is executed on LLVM execution engine to verify its functional correctness, and we also compare the runtime consumption 

### Methodology 

Take a simple Bril program as example. We take the JSON declaration of the program and have it analyzed by our LLVM IR generator. The overall workflow is similar to what we do in data flow analysis: 

  1. Create basic block mapping: traverse the bril IR in JSON representation, and form empty LLVM basic blocks according to block labels. We use a mapper from label string to LLVM basic blocks pointers. and also another pair struct to mark whether a basic block is used or not. 
 2. Insert instructions in to blocks: traverse the empty basic blocks and insert instructions into them. Each basic block should end with a valid terminating instruction (i.e. jmp, br or ret). The insertion process will exit after encountering the first terminator, and other following instruction under the same label will be ignored. 

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

 3. Dump LLVM code and run through the code with LLVM execution engine and compare the output with the ground truth produced by Bril interpreter. The execution engine will run the LLVM program with JIT. And we will evaluate the program's correctness with ground truth produced by Bril interpreter. 

### Implementing details

 In this section, we briefly describe the implementation details for each step in LLVM code generation process. To get a global knowledge of how different components are linked and executed in the program, we need a data structure recording of basic blocks and the actual values each instruction is taking. Here is a list of the data structure we created. Basically we categorize them into two classes: one is for block level and another for instruction level.

```c++
using BasicBlockFlag_T = pair<llvm::BasicBlock, bool> ;
using BasicBlockMap_T = map<string, BasicBlockFlag_T>;
using VarToVal_T = map<string, llvm::Value>;
```

The `BasicBlockFlag_T` is used to track whether a block has been visited or not. We will remove the unused basic blocks when traversing the whole program. Here followed a simple example of removing the redundant basic blocks

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

The `b2` branch will not be reached in any condition. During we insert the instructions, the block labelled `b2` will not be marked and we know that it is a redundant basic block. This basic optimization helps to reduce the executable size.

The `BasicBlockMap_T` structure helps construct a mapping from block name to actual LLVM blocks. When the instruction visitor traverses the control data flow graph, we create a new LLVM basic block every time we find a new label (except for the special case that entry basic block has no label). And the unordered mapping from label name to LLVM basic blocks will be created and saved for later use.

The `VarToVal_T` structure helps to track the actual value of each destination variable. And each instruction is allocated with their memory storage in the variable to value mapping. 

```c++
// create a alloca llvm value using IRBuilder
llvm::Value* val = builder->CreateAlloca(t_int_, llvm::ConstantInt::getSigned(t_int_, 1));
// save the allocated value into the VarToVal map
(*varToVal)[destination] = val;
```

Special note for the print instruction in Bril: we create a LLVM function call with integer return data type, and pass in `%d` and actual LLVM value to be printed as the arguments, and build a `CreateCall` node with LLVM IRBuilder so that the print function can be realized in LLVM program.

### Experiment Results

  To verify the correctness of the generated, we developed several test cases, which covers most commonly used arithmetic and control flow instructions, as well as some corner cases where the program has some redundant instructions that cloud have been removed. The test example is shown as followed:

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

Bril program's JSON declaration is generated with `bril2json`, by running command as followed, the JSON file will be generated and analyzed, then LLVM module will iterate the built-in LLVM functions (where basic blocks reside) and print out the LLVM program into the destination file.

```shell
cat test.bril | bril2json > test.json
./bril-llvm test.json test.llvm
```

By observing the generated LLVM code, we can find that at the very end of branch `b1`, a new instruction is added to avoid no terminator issue. and the print function of Bril is transformed into a LLVM function call with corresponding variables passed in as arguments.

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

After verifying the correctness of the code generator, we also compared the performance of LLVM simulation and Bril Interpreter. The performance is measured with profiling tool in Linux and C++. we run the same program for 10 times and take the average runtime. 
