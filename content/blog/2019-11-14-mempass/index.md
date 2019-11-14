+++
title = "MemPass"
extra.author = "Eashan Garg and Sameer Lal"
extra.bio = """
Eashan Garg is a senior undergraduate studying CS and English.

Sameer Lal is a CS Master of Engineering student. He studied ECE and Math in his undergraduate.
"""
+++

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

Once any of these vulnerabilities are detected, the program will throw a warning to the user. Upon completion of a program's execution, MemPass will generate a report listing the detected vulnerabilities in the original program.

We chose this design to allow us to see memory problems in real time during execution as well as afterwards. In practice, we can only detect memory leaks after the program has completely executed, while use-after-free and double-free errors can be detected during runtime and should result in the program halting to prevent subsequent malicious code from executing.

## Design Overview
As mentioned at the beginning, there is a large breadth of possible memory safety vulnerabilities. In order to tackle a modest subset of these bugs, our strategy of choice is a simple dynamic analysis pass over the LLVM IR.

MemPass inserts instrumentation after relevant memory allocation instructions, recording the relevant addresses. Anytime the program attempts to either access or deallocate memory, MemPass will then check if those addresses are still available for use. If not, the program will throw a warning.

> **Note**: For a more practical implementation, MemPass can easily choose to exit here and prevent any further malicious behavior, although we chose to emit a warning in order to make batch testing simpler.

In essence, if the program ever try to deallocate or access a memory address that isn't currently allocated in MemPass's hashtable, we have a bug!

## Instrumenting LLVM IR
To track memory allocations, deallocations, and accesses in LLVM IR, MemPass needs to insert instrumentation after relevant instructions.

While we could insert our own, carefully crafted LLVM instructions every time, we opted to write a runtime library and link it to the main program with our pass (as described [here](https://github.com/sampsyo/llvm-pass-skeleton/tree/rtlib)).

In this runtime library, we can write a series of functions to grab relevant data, and then perform the appropriate steps to detect any memory safety vulnerabilities. For each instruction, we log:

| Instruction | Logging |
| -------- | -------- |
| `alloca`     | size and (stack) pointer address|
| `malloc`     | size and pointer address |
| `calloc`     | size and pointer address |
| `free`       | pointer address |
| `load`       | address to load from |
| `store`      | address to store to |

Now, all we have to do is insert a `call` to one of our library functions after every relevant memory instruction, and the `llvm-link` tool will do all the heavy lifting!

## Tracking Memory Allocation
In order to better illustrate how MemPass works with a real LLVM program, lets take the following buggy program segment:

```c
int* ptr;
ptr = (int *)malloc(4 * sizeof(int));
free(ptr);
free(ptr);
```

We see a double free vulnerability, which we want to catch and emit as a warning to the user.

The relevant call to `malloc` in LLVM IR translates as follows:

```
%7 = call noalias i8* @malloc(i64 16) #3
```

To pass the relevant data to our logging functions, we need to grab both the memory address that `malloc` allocated, as well as the amount of memory that was allocated. Luckily, `%7` stores the pointer address as an 8-bit integer pointer, and the size is a 64-bit integer operand to the `malloc` call itself.

Thus, MemPass inserts a call to our `logMalloc` library function:

```
%7 = call noalias i8* @malloc(i64 16) #3
call void @logMalloc(i8* %7, i64 16)
```

Armed with this data, MemPass stores the address and memory size as key-value pairs in a hashmap. This allows us to easily check if an address has already been allocated.

Note that this approach will work similarly with any calls to `calloc`.

## Checking Free

Returning to the above example, the two calls to `free` roughly translate to (with instrumentation, after omitting a few loads and bitcasts):

```
call void @free(i8* %10) #3
call void @logFree(i8* %10)
call void @free(i8* %12) #3
call void @logFree(i8* %12)
```

In essence, MemPass sends the relevant addresses that we want to free to our runtime library. Taking the first address in `%10`, MemPass checks the allocation map to see if has been allocated. If so, this is a valid attempt to free!

On the other hand, our second call to `free` occurs attempts to free `%12`, which is the same address as `%10`. MemPass simply checks the allocation map, and since the address is no longer stored here we have a double free bug. MemPass prints this as a warning to the console, and continues searching for vulnerabilities.

While MemPass doesn't handle calls to `realloc`, it would be quite tractible to do so now. Just remove the old memory address from the allocation hash, and add the new address that the function returns (along with its size).

## Use After Free: Accessing invalid memory

One of the more difficult aspects of MemPass's implementation is to find a way to handle accesses to memory after a pointer has been freed. Consider the following program:

```c
int* ptr;
ptr = (int *)malloc(4 * sizeof(int));
free(ptr);
ptr[0] = 4;
return 0;
```

The naïve solution would be to add instrumentation after every load or store instruction in the LLVM IR, and compare the addresses to our allocation map. However, we run into a series of complications once we look at the actual IR.

```
%3 = alloca i32, align 4
%4 = alloca i32, align 4
%5 = alloca i8**, align 8
%6 = alloca i32*, align 8
store i32 0, i32* %3, align 4
store i32 %0, i32* %4, align 4
store i8** %1, i8*** %5, align 8
%7 = call noalias i8* @malloc(i64 16) #3
%8 = bitcast i8* %7 to i32*
store i32* %8, i32** %6, align 8
%9 = load i32*, i32** %6, align 8
%10 = bitcast i32* %9 to i8*
call void @free(i8* %10) #3
%11 = load i32*, i32** %6, align 8
%12 = getelementptr inbounds i32, i32* %11, i64 0
store i32 4, i32* %12, align 4
ret i32 0
```

First, the program accesses more memory than what was allocated by the programmer through calls to `malloc`. If MemPass compares the address accessed by one of the first `store` instructions with the allocation hash, it wouldn't find the address and emit a false-positive warning.

In reality, programs also allocates stack frame memory with `alloca` instructions. In order to handle these extra allocations, MemPass adds extra instrumentation here, and adds the stack frame addresses/sizes onto the allocation map.

However, there's another problem. Some of the load and store instructions use pointers of arbitrary types. If MemPass doesn't know what the pointer types are, it can't pass those addresses to the runtime library for evaluation.

A solution that MemPass employs is to insert `bitcast` instructions after every load or store instruction, converting the address pointer from its arbitrary type to an 8-bit integer pointer. Since we're only comparing addresses and not the actual values at these addresses, this should work somewhat well. However, the breadth of types in LLVM could allow for some imprecision here.

With `i8*` pointers, all MemPass needs is the size of the memory chunk that a load or store plans to interact with. While this data is not immediately accessible, LLVM provides a handy DataLayout class. After grabbing the type of the element that the *original* pointer points to, MemPass can extract its size and pass that to our library functions.

Finally, we need to actually check if we are accessing memory that is available as per the allocation hash. MemPass takes the difference of the address in question with every pointer address in the allocation map, and compares it to the appropriate sizes. If the address is within the bounds of a chunk of memory, then we are fine. Otherwise, the program will emit a use after free warning, and continue to look for more vulnerabilities.

> **Note**: These pointer address comparisons are somewhat sketchy, there likely may be a better approach to memory access checking that we were unable to find.

## Program Termination

On program termination, we need to check the allocation hash for any remaining memory that has not been freed. In order to differentiate stack memory that was allocated with `alloca`, any memory allocated with `malloc` is given an additional tag. Now, MemPass just searches the allocation hash and emits any malloced addresses that were not freed. This is compiled into a report, listing all of the memory safety vulnerabilities (among double free, use after free, and memory leak) that were detected throughout the execution of this program.

## Implementation extras

Another possible implementation scheme we considered was the use of a *deallocation* hashmap in addition to the allocation hashmap, to store memory that has been freed. This way, MemPass does not need add instrumentation after `alloca` instructions, it just needs to store memory addresses allocated with malloc. However, every time memory is allocated or deallocated, MemPass must check addresses in the other map to ensure there are no overlaps.

Both for the proposed framework and our current one, some sort of segmentation tree implementation to store memory bounds as intervals and compare them quickly. However, the overhead of building this tree might not be worth the benefits for small programs.

## Evaluation

When evaluating, we aimed to catch all the use-after-free, double-free, and memory leak errors that we could. We wrote a series of correctness tests first to verify that our algorithm worked as expected. Then, we selected a series of benchmark tests, checking both how many bugs MemPass was able to catch and the runtime overhead of our instrumentation.

Using a testing script, we take in a series of tests in `c`, and then compiled each one to LLVM IR using `clang` without any optimizations. We then link our runtime library and run MemPass, inserting instructions to keep track of memory usage. Finally, we use the LLVM IR interpreter, `lli`, to interpret the MemPass IR.  We collect the output reports and analyze all memory queries, looking for use-after-free, double-free, and memory leak errors.

### Correctness Testing
First, we wrote some simple test cases, both with and without the memory safety bugs we were looking for. These simple test cases involved allocating pointers and freeing them twice, pointer arithmetic, pointer aliasing and freeing multiple times after use.

While we tried to model these test cases off of the test cases other papers with similar designs used, we may have failed to capture some edge cases of our implementation (particularly with regards to use after free errors). Still, we were able to verify MemPass's correctness to a considerable degree.

### Benchmark Testing
We evaluated our design on subsets of the Toyota ITC and SARD-100 benchmark tests. The Toyota benchmark tests consist of a family of memory tests, and two test suites that we used are `Double Free` and `Memory Leak` tests.  The Sard-100 benchmark tests are similar, and we used the `cwe-415-double-free` and `cwe-416-use-after-free` suites.

In general, the`Double Free` and `cwe-415-double-free` benchmark tests consist of normal double free errors, freeing in constant/variable if statements, freeing in a function, freeing in conditional while loops, and freeing in for loops.

The `Memory Leak`, `cwe-401-memory-leak`, and `cwe-416-use-after-free` benchmark tests consist of a series of tests such as allocating memory without freeing, allocating in conditional statements, freeing based on function return values, allocating memory in mutually recursive functions and various branching scenarios.

A comprehensive description of each test case can be found on the respective pages: [Toyota ITC](https://github.com/regehr/itc-benchmarks) and [SARD-100](https://samate.nist.gov/SRD/testsuite.php#sardsuites).

### Benchmark Results
In [Detection of Security Vulnerabilities in C Code using Runtime Verification](https://nikolai-kosmatov.eu/publications/vorobyov_ks_tap_2018.pdf), the authors provide benchmark test results for E-ACSL, Google Sanitizer, and RV-Match on both the Toyota-ITC and SARD-100 test suites.

| Defect Type          | E-ACSL | Sanitizer | RV-Match |
|----------------------|--------|-----------|----------|
| Dynamic Memory Tests | 94%    | 78%       | 94%      |

Our optimization targed double free, memory leaks and use-after-free errors. So within the Dynamic Memory tests, we considered the `double-free` and `memory leak` suites of tests.  In total, this amounts to approximately 16 tests per suite.  Over these test suites we achieved the following results:

| Dynamic Memory Test | Double Free | Memory Leak |
|---------------------|-------------|-------------|
| MemPass             | 94%         | 100%        |

Unfortunately, MemPass is unable to detect the other memory bugs that the benchmark suite provides, so comparison to the other three memory bug detection tools may not be as useful here.

On the other hand, the paper provided a much more granular breakdown of test results on the Sard-100 test suite. The results, along with our results, are displayed below:

| Non-Memory Defects      | E-ACSL     | Sanitizer  | RV-Match   | MemPass   |
|-------------------------|------------|------------|------------|-----------|
| CWE-416: Use After Free | 100% (6/6) | 100% (6/6) | 100% (6/6) | 100% (6/6)  |
| CWE-415: Double Free    | 100% (6/6) | 100% (6/6) | 67% (4/6)  | 100% (6/6) |
| CWE-401: Memory Leak    | 100% (5/5) | 80% (4/5) | 60% (3/5) | 100% (5/5)

> **Note:** Our implementation achieves some sense of completeness, as we only emit a warning if a program tries to interact with memory that was not allocated to begin with. However, this dynamic analysis comes at the cost of soundness, especially since we only interact with one execution path at a time.

### Runtime Overhead
In order to analyze the runtime overhead of MemPass, we ran all of our benchmarks ten times each, and took the average of the wall clock times. We found almost no difference in the runtime of the programs before and after running MemPass, although this is presumably because the number of instructions in each were quite small.

Taking inspiration from the above benchmark paper, we wrote a small program that called `malloc`, `free`, and made simple memory accesses in a loop with one million iterations.

| Method | Average |
| -------- | -------- |
| Original Program     | 0.0757 seconds    |
| After MemPass        | 0.0887 seconds    |

Overall, we found that the overhead for MemPass caused a 1.17x slowdown on our small sample program. Still, this may not be representative of the actual overhead cost of MemPass, which should understandably be somewhat significant, as MemPass adds function calls after a large section of instructions in LLVM IR.

### Dynamic vs Static Analysis

While a dynamic analysis is certainly interesting and useful, it's difficult to analyze certain memory bugs such as memory leaks, since they cannot be detected until the program terminates. In addition, a dynamic analysis can only check individual executions of a program, and therefore might miss bugs with programs that have a large number of inputs.

A static analysis that uses some sort of use-def chain would be another interesting method to triage these vulnerabilities, and would relax much of the overhead that our method produces. In addition, it would be able to analyze all possible execution paths at once, and therefore complete a more 'sound' analysis of the various bugs that may be present in the program.