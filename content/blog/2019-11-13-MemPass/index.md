+++
title = "MemPass"
extra.author = "Eashan Garg and Sameer Lal"
extra.bio = """
Eashan Garg is a senior undergraduate studying CS and English.

Sameer Lal is a CS Master of Engineering student. He studied ECE and Math in his undergraduate.
"""
+++

# Dynamic Memory Profiling

While manual access to memory allocation/deallocation can allow the experienced programmer to significantly improve the performance of their program, it can also open the door to a host of memory safety bugs: use after free, double free, out of bounds memory accesses, and the like. Detecting these problems are difficult on the programmer's end and leave room for exploitation by malicious users.

## Memory Safety Vulnerabilities

Many memory safety vulnerabilities occur when the user tries to access or deallocate memory that isn't available for use, either because it was freed, or never allocated in the first place. Take for example the following program:

```c
int ptr*;
ptr = (int *)malloc(4 * sizeof(int));
free(ptr);
```

After allocating 16 bytes, the starting address for this block of memory is stored in `ptr`, and freed after use.

Now, assume that the programmer attempts to access the memory that was originally allocated by access 

```c
int *ptr, *ptr2;
ptr = (int *)malloc(4 * sizeof(int));
ptr[0] = 4;
free(ptr);
ptr2 = (int *)malloc(4 * sizeof(int));
printf("%d", ptr[0]);
```

Since `ptr` has been freed, the second call to `malloc` (which is of same size to the first) is likely to allocate the free space `ptr` points to. Now, lets assume a malicious attacker managed to store a dangerous payload in this space. Since the program tries to use `ptr` after it has been freed, it might actually end up accessing the attacker's payload, not the original `ptr` data as intended!

Double free errors occur upon consecutive `free()` call with the same memory address. For instance, consider the following code snippet taken from a Double-Free Toyota ITC test case:

``` c
int main() {
  char* ptr= (char*) malloc(sizeof(char));
  free(ptr);
  free(ptr); /* Double Free Error! */
}
```

Here, `ptr` is allocated and then freed twice.  Generally, double-freeing a block of main memory will corrupt the state of the memory manager, allowing resulting in a write-what-where condition.  A malicious attacker can write arbitrary dangerous code to that memory location.

Memory Leaks occur when the programmer forgets to free memory which is no longer needed.

```c
int main() {
    char* ptr= (char*) malloc(sizeof(char));
}
```

The consequences of memory leaks is often degraded performance.  Memory leaks reduce the available amount of memory in a system, and if not enough space is available, the program may slow down due to thrashing or stop altogether. 

## Introducing MemPass

Here, we present a analysis tool (creatively titled MemPass) for LLVM to detect use after free bugs, double free bugs, and memory leakage. 

Once any of these vulnerabilities are detected, the program will throw a warning to the user. Upon completion of a program's execution, MemPass will generate a report listing the detected vulnerabilities and their associated line numbers in the original program.

We chose this design to allow us to see memory problems in real time during execution as well as afterwards. In practice, we can only detect memory leaks after the program has completely executed, while use-after-free and double-free errors can be detected during runtime and should result in the program halting to prevent subsequent malicious code from executing.

## Design Overview
As mentioned at the beginning, there is a large breadth of possible memory safety vulnerabilities. In order to tackle a modest subset of these bugs, our strategy of choice is a simple dynamic analysis pass over the LLVM IR.

MemPass inserts instrumentation after relevant memory allocation instructions, recording the relevant addresses. Anytime the program attempts to either access or deallocate memory, MemPass will then check if those addresses are still available for use. If not, the program will throw a warning.

> **Note**: For a more practical implementation, MemPass can easily choose to exit here and prevent any further malicious behavior, although we chose to emit a warning in order to make batch testing simpler.

Our approach can be re 

### Instrumentation w/ rtlib

### Tracking Allocations
- Take an example
- Show how the algorithm stores

### Protecting Frees

### Accessing Freed Memory

### 

### Allocating Memory
As dynamically allocated memory by the programmer is our target for memory safety vulnerabilities, we need to identify malloc and calloc instructions, and then log the addresses that were allocated.

MemPass inserts function calls after every memory allocation, which grabs the addresses that these memory allocation instructions return, as well as the size of memory allocated. This data is stored in a hashtable, with the addresses as the keys and memory sizes as the values. This will allow MemPass to quickly detect whether a memory address is available to the program or not.

### Free
When memory is freed, we need to ensure that the requested address wasn't already freed, and then invalidate that address from within the allocation hashtable. MemPass extracts the appropriate address from the pointer that the program requests to free, then searches for that address within the allocation hashtable. If the address is not found, then MemPass will emit a 'Double Free' warning, as this means the memory address was freed beforehand. If the address is found, then MemPass will proceed to remove it from the allocation hashtable for future queries.

### Realloc
- Reallocating memory causes the sizes to change

### Loads/Stores
- Loads and stores are also instrumented
- Alloca instructions are instrumented to ensure that we only focus on malloc calls
- Pointers are cast to i32_t (Is this sound? Idefk) to compare with malloc

### Implementation
- Runtime library and instrumentation strategy
- Examples of code snippets that do this.

## Evaluation

In evaluation, we aimed to catch all use-after-free and double-free errors.  We performed a series of custom tests as well as benchmark tests to evaluate our algorithm.  We wrote a testing script that takes in a series of tests in `c` , and then compiled each one to LLVM IR using `clang` without any optimizations.  We then run MemPass on this Intermediate Representation, inserting instructions to keep track of memory usage.  Finally, we use the LLVM IR interpreter, `lli`, to interpret the MemPass IR.  We collect the output and analyze all memory queries, looking for use-after-free and double-free errors.  Because the end result consists of all memory queries, we are easily able to extend this to other memory analyses.  Finally, we generate a report on all test cases.

### Initial Testing
Initially we wrote simple test cases, some containing and some not containing memory-after-free and double-free errors.  These simple test cases involved allocating pointers and freeing it twice, pointer arithmetic, pointer aliasing and freeing multiple times after using.  We modeled these test cases off of the test cases other papers with similar designs used.  

### Benchmark Testing
We evaluated our design on subsets of the Toyota ITC and SARD-100 benchmark tests. The Toyota benchmark tests consist of a family of memory tests, and two test suites that we used are `Double Free` and `Memory Leak` tests.  The Sard-100 benchmark tests are similar, and we used the `cwe-415-double-free` and `cwe-416-use-after-free` suites.  

In general, the`Double Free` and `cwe-415-double-free` benchmark tests consist of normal double free errors, freeing in constant/variable if statements, freeing in a function, freeing in conditional while loops, and freeing in for loops.

The `Memory Leak` and `cwe-416-use-after-free` benchmark tests consist of a series of tests such as allocating memory without freeing, allocating in conditional statements, freeing based on function return values, allocating memory in mutually recursive functions and various branching scenarios.  A comprehensive description of each test case can be found on github.


### Benchmark Results
The below table compares the performance of our implementation against E-ACSL, Google Sanitizer and RV-Match on the Toyota ITC Benchmark tests and the sard-100 benchmark tests:

| Defect Type          | E-ACSL | Sanitizer | RV-Match |
|----------------------|--------|-----------|----------|
| Dynamic Memory Tests | 94%    | 78%       | 94%      |

Our optimization targed double free, memory leaks and use-after-free errors. So within the Dynamic Memory tests, we considered the `double-free` and `memory leak` suites of tests.  In total, this amounts to approximately 16 tests per suite.  Over these test suites we achieved the following results:

| Dynamic Memory Test | Double Free | Memory Leak |
|---------------------|-------------|-------------|
| MemPass             | 94%         | 100%        |

Data regarding the performance of other implementations are derived from the following [here](https://nikolai-kosmatov.eu/publications/vorobyov_ks_tap_2018.pdf).

Our implementation performed quite well on the Toyota ITC benchmark tests and performed modestly on the sard-100 tests.  For the Toyota ITC benchmark test, it suffices to note that the test we failed was due to nondeterministic input from a random number generator.

For the Sard-100 test suite, we achieved the following:

| Non-Memory Defects      | E-ACSL     | Sanitizer  | RV-Match   | MemPass   |
|-------------------------|------------|------------|------------|-----------|
| CWE-416: Use After Free | 100% (6/6) | 100% (6/6) | 100% (6/6) | 100% (6/6)  |
| CWE-415: Double Free    | 100% (6/6) | 100% (6/6) | 67% (4/6)  | 100% (6/6) |
| CWE-401: Memory Leak    | 100% (5/5) | 80% (4/5) | 60% (3/5) | 100% (5/5)





We achieved our goal of completeness.  Naturally, since we detect memory bugs at runtime, none of the detected leaks in our generated report are false positives.


  
### Extensions and Improvement

We ideally would like to implement an initial static analysis sweep that flags certain instructions that may result in a memory leak. One main drawback of our dynamic approach is that we are only testing for memory leaks on one execution of the program, and since programs can be nondeterministic, we may miss memory leaks that occur in subsequent executions.  Static analysis can help catch tag errors in branches that our program during runtime may not hit.  We could then the program multiple times until we hit each tagged instruction or reach desired coverage.  

We would also like to optimize the instructions in the LLVM IR that we insert in order to do our memory analysis.  Currently, running IR code before MemPass and after MemPass realizes a noticable change in runtime.  Ideally, a possible extension to this project would be to try to lessen overhead while still maintaining the same functionality. 

We did run into troubles with logging malloc instructions that allocated very few amount of bytes.  Upon testing, we did not catch several use-after-free errors which we attribute due to LLVM promoting a few `malloc` instructions to `alloca` instructions.  This makes it difficult to distinguish whether we are accessing the freed pointer itself or the unfreed cached version of the pointer.  In these cases, we assert that the operation is safe, as we do not want to output false positives. We hope to research ways to prevent



