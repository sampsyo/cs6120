+++
title = "A Backend That Translates Bril into C"
extra.author = "Wen-Ding Li"
extra.author_link = "https://www.cs.cornell.edu/~wdli/"
extra.bio = """
  [Wen-Ding Li](https://www.cs.cornell.edu/~wdli/) is a Ph.D. student at Cornell interested in high-performance code generation.
"""
+++

Bril is an educational compiler intermediate representation that is designed for this compiler course. While there is already a Bril interpreter written in
TypeScript called `brili`, it would be interesting to have other backends so that we can compare the performance and the implementation complexity. In this first project,
I built a backend that translates Bril into the C language and then use GCC to compile and execute the program.

Why choose C?
---
Translating to C provides some benefits such as portability, good performance, and easier integration with other C library and tools. The C language is a widely used language in many embedded devices. We can also get native performance and leverage the GCC compiler's optimizations. By translating to C, we can easily integrate it with other C libraries. Plus, because the Bril instructions (as it is now) can be mapped to C statements, we can also potentially use gdb as a debugger.
Lastly, translating to C is very common in the programming community and can be a fun project!

Method
---
As of now, Bril only has one main function. For the sake of simplicity, I first collect all the name and type of variables used in the Bril program, and declare them at the top of the generated C code with the corresponding type, where I use `int64_t` for `int` and `int` for `bool`. Then, all the arithmetic, comparison, and logic instructions can be directed translated to corresponding C statement. For label and `jmp` instructions, I use `label` and `goto` in C. The instruction `br` can also be easily translated to statement compose of `if` and `goto`.

In order to verify the correctness of this C backend, I created some valid handwritten tests to verify against the existing interpreter `brili`. The C backend successfully passes all the tests.

Here we show a small example to demonstrate the translation.
```
main {
  a: int = const 4;
  b: int = const 4;
  cmp: bool = ge a b;
  jmp somewhere;
  a: int = const 2;
  somewhere:
  c: int = add a b;
  print c;
  print cmp;
}
```
The above Bril program can be translated to C as the following.
```C
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
int main(){
int64_t a;
int64_t b;
int cmp;
int64_t c;
a = 4LL;
b = 4LL;
cmp = a >= b;
goto somewhere;
a = 2LL;
somewhere:;
c = a + b;
printf("%" PRId64 "\n", c);
printf(cmp?"true\n":"false\n");
return 0;
}
```

Implementation
---
I implemented the translation tool in Python. The implementation is straightforward and consists of 134 lines of code.  Compared to other projects such as [codegen-llvm](https://github.com/seanlatias/bril/tree/master/codegen-llvm) for Bril using LLVM that consists of 500+ lines of C++ code and [bril2jb](https://github.com/Neroysq/bril2jb) generating Java bytecode consisting of 400+ lines of Java code, I think it is safe to say that the implementation complexity of C backend is smaller.

The source code and tests can be found at [Bril2C](https://github.com/xu3kev/bril2c). The tool to translate Bril JSON format to C is `bril2c.py`. It takes input on stdin and produces output on stdout.

Benchmark
---
I created some simple benchmarks to measure the timing compared with the interpreter run in Node.js.

Each test was measured while running 1000 times. In order to avoid the startup time of Node.js, I slightly modified the `brili` interpreter to run the Bril program 1000 times internally so I can avoid running `brili` 1000 times.

The experiments are all run on Intel Xeon CPU E5-1630 v3 @ 3.70GHz with Ubuntu 16.04. Turbo boost is turned off and the scaling governor is set to performance.

The version of Node.js to run Bril interpreter is 12.11.0.
The gcc version  is 5.4.0 and the optimization flag is O3.

The benchmark result is as following.
<br>
<img src="c_backend_benchmark.png" width="500">
<br>
The four test programs are factorial computation, the Fibonacci sequence, polynomial multiplication and matrix multiplication.
We can see that we gain a significant speedup across different tests. However, because Bril now has only bool and int type, the things we can compute are still limited. In the future, as Bril extends with more features such as floating point arithmetic and arrays, we can have more practical benchmarks.

Conclusion
---
In this project, I built a C backend for Bril. I verified its correctness and benchmarked its performance on several tests. The result shows that compared with the `brili` interpreter, It gains a significant speedup.

Acknowledgment
---
I want to thank Hongbo and Siqiu for the helpful discussion.  I also want to thank Adrian and Matthew for the feedback on this project.
