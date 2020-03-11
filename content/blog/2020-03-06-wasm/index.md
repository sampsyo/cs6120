+++
title = "Bringing you up to speed on how compiling WebAssembly is faster"
extra.author = "Sachille Atapattu"
extra.author_link = "https://www.csl.cornell.edu/~sachille/"
extra.bio = """
  [Sachille][] is a third-year graduate student working across the stack to 
  make heterogeneous computing easier. Sachille also likes books (currently reading 
  Howards End by E.M. Forster) and philosophizing with friends.

[sachille]: https://www.csl.cornell.edu/~sachille
"""
+++

If you're a geography buff like me (you're probably not, but one can hope), 
You might remember the time when you downloaded an app to get the exhilrating
feeling of zooming on top of a volcano, or to an island in the middle of the 
Pacific, or to the top of the Eiffel. Then do you remember when you 
downloaded Chrome just you can do the same from the convenience of the browser? Well, 
then you might have just heard about how you can do it in any browser out there now. 
For me, that's what WebAssembly brings.

In this article, I don't expect to talk much about the language (or binary code format) 
itself, but implications of its compilation which makes 
things like Google Earth on a browser possible.

<img src="sys-diagram.png" width="700" >

## What is WebAssembly
WebAssembly is a binary code format to transfer web applications from the 
server to the browser. It is incorporated in modern browsers to be used in 
tandem with existing JavaScript applications, and uses existing JavaScript 
engines to interpret and execute. It has taken the world wide web by storm a
- it is rolled out by four major browsers (Chrome, Edge, Mozilla and Safari)
i.e. platform independent
- it is programming model independent
- it is hardware independent

What this means is you could write a web application in C, compile it to 
WebAssembly and use it on any browser on any hardware. This gives much more 
performance in general than a JavaScript program, which has type 
classification overheads, performance implications based on which browser 
you target and how performant is your interpreter.

(Why can't you run C in the first place?)

## Compilers for WebAssembly
Major component of speed up from WebAssembly comes from the compilation.
Using JavaScript, your JavaScript engine would go through the phases of 
parsing, baseline compilation, optimizing compiler, re-optimizing and 
bail out, execute and garbage collection to run an application. 
WebAssembly affects each of these stages to be more performant.

To begin with, WebAssembly is more compact than JavaScript source code, 
making it faster to fetch from the server. Then WebAssembly doesn't need 
parsing, it's already compiled down to virtual instructions which only 
need decoding like in actual hardware. The engine can do this decode 
much faster than parsing JavaScript code to an IR.

Then benefits of WebAssembly from actual compilation kick in. 
JavaScript needs to be compiled to multiple versions based on what types 
are in use(similar to any other dynamically typed language). WebAssembly 
code has its types encoded during offline compilation on to WebAssembly.
Therefore, it doesn't need monitoring (in the interpreter) to figure out 
the types, and maintain multiple versions. The JavaScript engine also 
doesn't need to do most optimizations, except for platform and hardware 
dependent ones, as everything else is already done in static compile time.

Since WebAssembly doesn't need assumptions (such as which type certain 
object is) during interpretting, the JavaScript engine doesn't  need to 
bail out and reoptimize as such errors never occur. 

Finally, WebAssembly also allow you to manage memory manually (it only 
suports manual memory management as of now, but automation is to be added
 as an option) which allows you to avoid expensive garbage collection 
during interpretation. 

## Evaluation
Writing code in WebAssembly doesn't mean it'll be automatically faster. 
JavaScript can be in theory, more performant in execution (at least for now, 
where WebAssembly is interpretted using JavaScript enging). But this 
requires the programmer to know JIT compilation internals and constrain to 
one browser as each has different interpreters. WebAssembly is
already optimized statically (more time for the compiler to optimize) for 
the general case and each interpreter can leverage it's generality to do 
additional optimizations better than on generic JavaScript. So in practice
WebAssembly can be much more performant.

(Add numbers from the paper)


## What this means

## What's next?
WebAssembly has already achieved a lot in a short span of time (3 years 
since introduced in 2017). It is already supported in all major browsers 
and popular development kit Emscripten offers compiling C down to `.wasm`
via `asm.js`. Personally, it has already enabled running Google Earth on 
any browser (what more do you need!!).

Yet, WebAssembly is at its infancy. The popular compiler toolchain, 
LLVM project is adding a backend for it and developers are adding debug tools
to improve WebAssembly eco-system.

More exciting and enabling future options to WebAssembly may include 
- stream compilation: start compiling as the byte code is being downloaded, 
- shared memory concurrency: reduce synchronization by handling it 
efficiently on shared memory and 
- SIMD: to parallelize execution by sharing instructions among data. 

