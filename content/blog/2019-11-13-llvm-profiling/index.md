+++
title = "Runtime Execution Profiling using LLVM"
extra.author = "Yi Jing, Zhijing Li, Neil Adit"
extra.bio = """
[Zhijing Li](https://tissue3.github.io/) is a 2nd year Phd student at ECE department, Cornell. She is interested in compiler optimization and domain-specific languages. \n
[Yi Jiang](http://www.cs.cornell.edu/~yijiang/) is a 2nd year Phd student in CS dept, Cornell. She is interested in designing compilers for Processing-In-Memory related architectures and also general compiler optimization. \n
[Neil Adit](http://www.neiladit.com) is (guess what!) also a 2nd year PhD student at CSL, Cornell. He is interested in fun things in compiler research.

"""
extra.latex = true

+++



## Goal

The goal of the project is to collect run-time information by adding an LLVM pass that is accurate even in multi-threading program. We are interested in three kinds of information: 

1. Basic execution information including the number of LLVM instructions, basic blocks etc.
2. Expensive instruction information like the number of memory operations, multiplications and branches since they are most likely to affect execution time.
3. Function call information: the number of times each function is called.



## Design Overview

The LLVM architecture looks like this:

<img src="llvm.jpg" style="width: 90%">



"IR" stands for *intermediate language*. We only need to add one pass that changes the existing IR program to another IR program (program which includes the desired properties like profiling information), following instructions to add an LLVM pass in our instructor, [Adrian](<https://www.cs.cornell.edu/~asampson/>)'s [blog post](<https://www.cs.cornell.edu/~asampson/blog/llvm.html>). We also took help from the Github repo [atomicCounter](<https://github.com/pranith/AtomicCounter/blob/master/AtomicCountPass/AtomicCount.cpp>) where the increments to the global counter variables are atomic updates

We start from [ModulePass](<http://llvm.org/docs/WritingAnLLVMPass.html#the-modulepass-class>) and then access functions inside a Module, BasicBlocks per function and instructions per BasicBlock to obtain all information. At the beginning and ending of each module, we also call a custom method `initialize` to create global variables in IR  and `finalize` to print global variables. The structure of our atomic counting pass looks like this:

```c++
  struct SkeletonPass : public ModulePass {
    static char ID;
    SkeletonPass() : ModulePass(ID) {}
      
    virtual bool runOnModule(Module &M); //when there is a Module
    virtual bool runOnFunction(Function &F, Module &M); //called by runOnModule
    virtual bool runOnBasicBlock(BasicBlock &BB, Module &M); // called by runOnFunction
      
    bool initialize(Module &M); //create global variable
    bool finialize(Module &M); //print global variable
      
    void createInstr(BasicBlock &bb, Constant *counter_ptr, int num);
      
    vector<string> atomicCounter; //keep global variable names for profiling. e.g. instr counter
  };
```



## Implementation

### Basic Execution Information

We first create global variable `llvmInstrAtomicCounter` and `basicBlockAtomicCounter` with [GlobalVariable](<https://llvm.org/doxygen/classllvm_1_1GlobalVariable.html#a3ef813d6bda7e49e31cb6bf239c4e264>) constructor.

```c++
new GlobalVariable(M, I64Ty, false, GlobalValue::CommonLinkage, ConstantInt::get(I64Ty, 0), atomicCounter[i]);
```

 Then in each basic block we obtain a pointer to the global variable names with [getOrInsertGlobal](<https://llvm.org/doxygen/classllvm_1_1Module.html#abd8f7242df6ecb10f429c4d39403c334>) method. After getting the instruction number in each block, we create atomic addition with [AtomicRMWInst](<https://llvm.org/doxygen/classllvm_1_1AtomicRMWInst.html#abf7e0649c7f272cc49165e579be010a5>) constructor. The complete code looks like this:

```c++

void SkeletonPass::createInstr(BasicBlock &bb, Constant *counter_ptr, int num){
    // create atomic addition instruction
    if(num){
        new AtomicRMWInst(AtomicRMWInst::Add,
                      counter_ptr, // pointer to global variable
                      ConstantInt::get(Type::getInt64Ty(bb.getContext()), num), //create integer with value num
                      AtomicOrdering::SequentiallyConsistent, //operations may not be reordered
                      SyncScope::System, // synchronize to all threads
                      bb.getTerminator());//insert right before block terminator
    }
}

bool SkeletonPass::runOnBasicBlock(BasicBlock &bb, Module &M)
{
    // get the global variable for atomic counter
    Type *I64Ty = Type::getInt64Ty(M.getContext());
    Constant *instrCounter = M.getOrInsertGlobal("llvmInstrAtomicCounter", I64Ty);
    assert(instrCounter && "Could not declare or find llvmInstrAtomicCounter global");
    Constant *bbCounter = M.getOrInsertGlobal("basicBlockAtomicCounter", I64Ty);
    assert(bbCounter && "Could not declare or find basicBlockAtomicCounter global");
    // get instruction number and basic block number.
    int basic_block = 1;
    int instr = bb.size();
    // create atomic addition instruction
    createInstr(bb, bbCounter, basic_block);
    createInstr(bb, instrCounter,instr);
    
    return true;
}

```

Finally, we print the profiling results with global variable names and the corresponding values. The values are obtained with a `load` instruction. Right before `return` of the last block in `main`, we create string variable corresponding to variables we are tracing and get the pointer to that string variable. Also, we get the number variable by first getting the pointer pointing to the global variable and then creating a load instruction to get the actual pointer to the value. Finally, we can call the `printf` function with input arguments.

```c++
    IRBuilder<> Builder(M.getContext());
    Function *mainFunc = M.getFunction("main")
    // not the main module
    if (!mainFunc)
        return false;

    // build printf function handle
    std::vector<Type *> FTyArgs;
    FTyArgs.push_back(Type::getInt8PtrTy(M.getContext())); // specify the first argument, i8* is the return type of CreateGlobalStringPtr
    FunctionType *FTy = FunctionType::get(Type::getInt32Ty(M.getContext()), FTyArgs, true); // create function type with return type, argument types and if const argument 
    FunctionCallee printF = M.getOrInsertFunction("printf", FTy); // create function if not extern or defined
    
    for (auto bb = mainFunc->begin(); bb != mainFunc->end(); bb++) {
        for(auto it = bb->begin(); it != bb->end(); it++) {
            if ((std::string)it->getOpcodeName() == "ret") {
                // insert printf at the end of main function, before return function
                Builder.SetInsertPoint(&*bb, it);
                for(int i = 0; i < atomicCounter.size(); i++){
                    Value *format_long = Builder.CreateGlobalStringPtr(atomicCounter[i]+": %ld\n", "formatLong");// create global string variable formatLong, add suffix(.1/.2/...) if already exists
                    std::vector<Value *> argVec;
                    argVec.push_back(format_long); 
                    Value *atomic_counter = M.getGlobalVariable(atomicCounter[i]);// get pointer pointing to the global variable name
                    Value* finalVal = new LoadInst(atomic_counter, atomic_counter->getName()+".val", &*it);// atomic_counter only points to a string, but we want to print the number the string stores
                    argVec.push_back(finalVal); 
                    CallInst::Create(printF, argVec, "printf", &*it); //create printf function with the return value name called printf (with suffix if already exists)
                }
            }
        }
    }
    return true;
}
```



### Expensive Operations Information

This part follows the same flow as basic execution information except we need to distinguish the instruction type and increment corresponding counter on a block basis. Therefore in each `runOnBasicBlock` method, we need the following lines:

```c++
    for (auto it = bb.begin(); it != bb.end(); it++) {
        switch (it->getOpcode()) {
            case Instruction::Mul:// multiplication
                mul_instr++;
                continue;
            case Instruction::Br:// branch
                br_instr++;
                continue;
            case Instruction::Store:// store
            case Instruction::Load:// load
                mem_instr++;
                continue;
            default:
                break;
        }
    }
```

### Function calls

We profile the number of times each function was called in the program. This is done by first initializing global variables corresponding to all the functions in the program to zero. This is done statically by iterating over a function list in the module given by `getFunctionList()`. This returns all the functions called in the program including the ones that were included from C libraries like `printf` or `scanf`. 

```c
auto &functionList = M.getFunctionList(); // gets the list of functions
for (auto &function : functionList) { //iterates over the list
    Value *atomic_counter = new GlobalVariable(M, I64Ty, false, GlobalValue::CommonLinkage, ConstantInt::get(I64Ty, 0),  function.getName()+".glob"); // create a global variable, name it based on the function name
} 

```

Next we want to insert atomic counters at the start of each function definition. This ensures that irrespective of multiple return points in the function we can always increment the counter for it at the beginning. We start with the entry basic block in the function which is given by the iterator `F.begin()`.  To insert it at the top of the basic block we use `getFirstNonPHI()` which returns the first instruction that is not a PHI node. We insert an atomic add instruction similar to other profiling instructions. 






## Hardest Parts

1. For people who are new to LLVM, instructions are hard to find and follow. Searching on Google can help if you know what you're looking for. It's difficult to get helpful information unless your search is confined to existing functions. Even though LLVM documentation is pretty exhaustive, it has too many functions to go through and the lack of examples can be off putting. Tutorials and existing backbone codes on Github can be really handy in these scenarios, which we took advantage of. It not only helped us implement specific functions like `printf` but also establish a structure to our IR pass. 
2. String manipulations: I am not sure if this is the right term to use, but LLVM seems to have 2 string types - Twine and StringRef. `getName` on a function returns a StringRef. In order to make a custom name I perform `F.getName()+"name"` which returns a Twine. But the function `getGlobalVariable` only accepts StringRef. Twine has a function `str` which can be used to convert it into string. Even though this is a straightforward solution, it ended up taking time to figure out the problem and looking into the documentation of these classes.
3. Setting up and running benchmarks: "It's all fun and games until you run your IR pass on real programs" - anonymous. We faced issues setting up benchmarks like PARSEC on Mac. Embench had multiple source files to compile which ran into trouble partly due to our IR pass not being thoughtfully written. We were defining global variables in all the files irrespective of it being a function/utility file or the main source file. We ended up using [Phoenix](https://github.com/kozyraki/phoenix) which worked well on Linux but was not meant for Mac. Hence for doing our LLVM pass, we had to install and update libraries on Linux machines.



## Evaluation and results

### Gcov profiling tool

To validate and benchmark our profiling results, we use `gcov` testing tool which can be used as a profiler to give performance statistics. We first compile the code using gcc flags required for `gcov` : `gcc -fprofile-arcs -ftest-coverage foo.c` . Now we run `gcov` with relevant flags to give us statistics to compare with our profiler:
```
gcov -b -c -f foo.c
```
We ran the [Phoenix](https://github.com/kozyraki/phoenix) benchmark suite and used `gcov` to profile statistics of function calls. The makefile initially had optimization flag -O3 but this might lead to incorrect numbers of function calls due to optimizations, so we compile without any flags. We picked `Kmeans` arbitrarily to demonstrate a detailed example of our profiling tool below:

#### Kmeans

Sequential execution without optimization flags yields the following output for function calls by `gcov`:

```
main: 1
dump_matrix: 1
calc_means: 23
find_clusters: 23
add_to_sum: 23000
get_sq_dist: 230000
generate_points: 2
parse_args: 1
```
We only list the function calls from the profiling tool `gcov` for sanity check. Our profiling pass also outputs other instruction statistics. The output generated by our profiling pass is listed down below:

```
llvmInstrAtomicCounter: 40091156
basicBlockAtomicCounter: 5019598
mulAtomicCounter: 691267
memOpAtomicCounter: 20250888
branchAtomicCounter: 4766547

parse_args: 1
generate_points: 2
add_to_sum: 23000
find_clusters: 23
get_sq_dist: 230000
calc_means: 23
dump_matrix: 1
main: 1
```

The numbers match with `gcov`. We compile the results for all benchmarks below.

### Results

The following results are for sequential execution of the benchmarks. We have reported the instruction counts in the following table:

| Benchmark | LLVM instruction count |basic block count | multiplication count | memory operation count | branch operation count|
| -------------- | :------------------------: | :------------------: | :--------------------: | :---------------------------: | :------------------------: |
| Histogram | 1707337893 |  104532503 | 0 | 871090234 | 104532501 |
| Kmeans | 40091156 |  5019598 | 691267 | 20250888 | 4766547 |
| Linear regression | 2244735757 |  86335992 | 86335983 | 1093589204 | 86335991 |
| PCA | 35388 |  3461 | 573 | 17975 | 3454 |
| String match | 3652012341 |  454936402 | 0 | 1575281754 | 443891533 |
| Word Count | 1148215642 |  213577613 | 46026 | 512924102 | 199431461 |

Function counts for the benchmarks are listed below (they have been matched with `gcov`s output as well):

Histogram:
```
test_endianess: 1
main: 1
```

Kmeans:
```
parse_args: 1
generate_points: 2
add_to_sum: 23000
find_clusters: 23
get_sq_dist: 230000
calc_means: 23
dump_matrix: 1
main: 1
```

Linear Regression:
```
main: 1
```

PCA:
```
parse_args: 1
dump_points: 2
generate_points: 1
calc_mean: 8
calc_cov: 8
main: 1
```

String match:
```
getnextline: 5522432
compute_hashes: 5522435
string_match: 1
main: 1
```

Word Count:
```
wordcount_cmp: 579521
wordcount_splitter: 1
wordcount_getword: 1
wordcount_addword: 1513425
dobsearch: 1513425
main: 1
```

### Pthread execution

We also ran `kmeans` using pthreads on `8` threads. We can see that some of the function calls in the output are scaled by 8 in both `gcov`  and our profiling pass. Matching results also show that atomic counters were successfully implemented.

`gcov` output:

```
main: 1
calc_means: 184
find_clusters: 184
add_to_sum: 23000
get_sq_dist: 230000
generate_points: 2
parse_args: 1
dump_points: 1
```
Our profiling output:

```
llvmInstrAtomicCounter: 40125466
basicBlockAtomicCounter: 5023832
mulAtomicCounter: 691429
memOpAtomicCounter: 20266988
branchAtomicCounter: 4770459

dump_points: 1
parse_args: 1
generate_points: 2
add_to_sum: 23000
find_clusters: 184
get_sq_dist: 230000
calc_means: 184
main: 1
```


