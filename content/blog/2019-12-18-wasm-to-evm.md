
Ethereum has developed a stack-based architecture for executing their own Turing-complete instruction set called Ethereum Virtual Machine (EVM). The EVM currently used for running decentralized applications on Ethereum blockchain. 
The smart contract code is usually written in higher-level languages, such as Solidity. The code must be compiled into bytecode before it can be run on EVM. 
EVM has 256-bit words, which are suitable for various cryptograph-related operations in smart contract applications. On the other hand, it makes it more difficult to translate EVM bytecode to hardware instructions and run the EVM program natively. 
To improve the performance of smart contract applications, Ethereum Web Assembly (Ewasm) was proposed to substitute the EVM in future Ethereum. Ewasm is a modified version of Web Assembly with some Ethereum related functions. 
Another advantage of Ewasm is that LLVM has a Wasm backend, so that smart contract applications developed in other higher-level languages, like C++ and Rust, can be compiled to Ewasm directly. Developers can use the languages they are familiar with instead of learning Solidity. 
However, it is still not clear when Ewasm development will be finished, so currently, developers still need to learn Solidity to write smart contract applications. It would be interesting if it is possible to provide a "forward compatibility" for EVM so that developers can start developing new smart contract applications in other higher-level languages, like C++ and Rust. The program will be compiled into EVM bytecode and run on the current Ethereum blockchain. Once Ewasm is online, the applications could be easily migrated to the new system.
One approach is to develop an LLVM backend to EVM. This project is going to investigate another approach by binary translation. 

## Binary Translator 

Since the Ewasm standard is not finalized yet, there is no existing compiler for Ewasm, so this proof-of-concept project will focus on translating a single computation-based function from Wasm to EVM. In the following discussion, we only consider the bytecode of the function and ignore other wrapping code.

For example, given a simple C function "add", 
```C
int add(int a, int b) {
    return a + b;
}
```
it can be compiled to Wasm bytecode `200120006a`.
```
...
20    ; get_local
01    ; $01
20    ; get_local
00    ; $00
6a    ; i32.add
...
```
Wasm has a stack-based design, so the code means

1. push parameter $1 onto the stack
2. push parameter $0 onto the stack
3. pop the top two elements from the stack, add them, and push the result back to the stack

This can be translated to EVM bytecode

```
...
6024    ; PUSH 0x04
35      ; CALLDATALOAD load second parameter at 0x04
6004    ; PUSH 0x24
35      ; CALLDATALOAD load first parameter at 0x24
01      ; ADD add top two elements on stack
...
```

Since both architectures are stack-based, the implementation of the basic binary translator is straight-forward. Many logic and arithmetic operations can be translated directly according to the opcode table. Here I will discuss some interesting optimizations instead. 

In EVM, there is a concept called "gas", which is a metering unit for measuring the cost of each instruction. The gas price is calculated by electricity and storage used for running the instruction. The smart contract users need to pay the price to the miners who run the smart contract application. 
Unfortunately, the gas price for Ewasm instructions is not out yet, we cannot compare the total cost for running an Ewasm program and a translated EVM program, but we can still optimize EVM bytecode to reduce the gas price of a function.

In Wasm, function parameters are passed by local variables. They are accessed by `get_local` instruction followed by an index number. Similarly, function parameters are stored in memory, which can be accessed by memory load operation `MLOAD` with an offset. For example, in the above function with two uint256 parameters, the first parameter is at 0x04, the second parameter is at 0x24. 
Wasm compiler tends to use the local variable like registers, the compiled bytecode will load data by `get_local` whenever it needs it. However, this approach is not optimal for EVM, especially when the program needs to load a parameter multiple times in a short period. 
For example, `if (a > b) return a - b` needs to load both "a" and "b" twice, 
one for `a > b`, and one for `a - b`. In EVM, it can load both parameters for the first time, so both parameters are currently first and second elements on the stack. It duplicates both parameters by calling `DUP2` twice, comparing to loading the data from memory, this approach has a lower cost. 
```
| |         | |         | |
| |         | |         |a|
| | -DUP2-> |b| -DUP2-> |b|
|a|         |a|         |a|
|b|         |b|         |b|
```
EVM provides a set of `DUP` instructions, it can duplicate one of 1~16th stack item and push it on the top of the stack. 

Currently, the binary translator can translate simple Wasm functions to EVM bytecode. 

While Wasm code can be easily run locally with NodeJS, I did not figure out how to run EVM code locally for benchmark.

## Some Thoughts

After working on this project, I found it is possible to translate Wasm programs to EVM bytecode to provide "forward compatibility" for Ethereum. At least, it can provide partial "forward compatibility" so that developers can write some core computation functions in languages like C++, compile to Wasm, and translate to EVM. In this way, the developer can write the function in languages they are familiar with and utilize existing compiler optimizations of those popular languages. 
However, I found binary translation might not be an optimal way to do this. Even there are many similar designs in both EVM and Wasm, it cannot achieve an optimal code without understanding the higher-level meaning of the code. Therefore, I think using intermediate representation (IR) might be a better solution, as the ongoing LLVM-EVM project. The Ethereum community is also developing an Ethereum  IR call Yul, which will be able to compile both EVM and Ewasm code. 
