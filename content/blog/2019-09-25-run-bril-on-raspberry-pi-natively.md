+++
title = "Run Bril on Raspberry Pi Natively"
extra.author = "Hongbo Zhang"
extra.author_link = "https://www.cs.cornell.edu/~hongbo/"
extra.bio = """"""
+++

[Bril][] comes with a reference interpreter `Brili`, 
which allows any platform that supports `Node.js` to run Bril code.
The downside of using an interpreter is that
the performance is usually not as good as a native binary.
Therefore, it would be more efficient (and cool) if Bril could be run natively on a ARMv8 processor.
In this project, I am going to build a code translator that
translate Bril to AArch64 assembly,
so that it could run on a 64-bit Raspberry Pi (Raspberry Pi 3B or later versions)
or any other 64-bit ARM devices with AArch64 architecture.

[bril]: https://github.com/sampsyo/bril

## Design and Implementation

This section will discuss how the translator is designed and implemented.
Although there are almost one-to-one mappings between Bril instructions and AArch64 assembly,
there are still some details that needs to be carefully designed to support
current and future functions of Bril.

### Types

Currently there are two values types in Bril:

* `int`: 64-bit two's complement signed integer type. 
It is equivalent to the `int64_t` type in C.
It will occupy 8 byte of memory and fit in a single 64-bit register of 64-bit ARM processors.
* `bool`: Boolean value that could be either `true` or `false`.
It is equivalent to the `bool` type in C.
It will occupy one whole byte in memory. 

### Variables and Stack Allocation

Currently, Bril only has local variables with the scope of entire function.
Similar to the local variables in C, all local variables are stored on the 
stack frame of the current function.

```
|--frame pointer---|
| local variables  |
|------------------|
| callee-save regs |
|------------------|
```

Unlike C, where variables needs to be explicitly declared, 
Bril instruction with a `dest` opcode will inexplicitly declare a variable.
In order to build the symbol table and allocate stack space for all variables,
it needs to scan all instructions in the function and add all `dest` to 
the symbol table.

ARMv8 requires that the stack pointer is 16-byte aligned.
One easy solution for stack allocation is that each variable occupies one
16-byte stack frame.
However, this is very inefficient since current supported types have size of
8 bytes or less.

The better solution is to keep track of fragmented space on stack. 
When adding a new variable to the stack, 
it will check if there is any fragmented stack space is big enough for the 
variable.
It will only increase the stack size by 16 bytes if there is no enough space.

For example, the following Bril program  

```
main {
    a:int = const 1;
    b:bool = const true;
    c:int = const 1;
    d:bool = const false;
}
```
will have stack allocation like this

|variable|offset|
|--------|------|
|a       |0x0000|
|b       |0x0008|
|c       |0x0010|
|d       |0x0009|

with free stack space `0x000A~0x000F` and `0x0018~0x001F`.

### Program and Functions

Each Bril file contains one program, which is the top-level object.
Therefore, each Bril file can be translated into one AArch64 assembly file.
Similarly, `main` function is the entry point for the program.

A function in AArch64 assembly consists a label as the function name and a sequence of instructions.
```assembly
func-name:
    instr1
    instr2
    ...
```

At the beginning of a function, it will first push all callee-save registers
on to stack, including frame pointer (`x29`) and link register (`x30`).
Then will build symbol table for current function 
and move the stack pointer and frame pointer accordingly to leave enough space
for all variables.

At the end of a function, there is a label indicating the return point.
Before `ret` to the address in link register, 
it needs to pop out all local variable by moving the stack pointer back,
   and restore all saved register values.

With this design, function call could be added easily with minor changes
for passing parameters.

### Arithmetic and Logic Operations

Arithmetic, logic, and comparison operations are easy to translate.

|Bril |AArch64|
|-----|-------|
|`add`|`add`  |
|`sub`|`sub`  |
|`mul`|`mul`  |
|`div`|`sdiv` |
|`and`|`and`  |
|`or` |`orr`  |
|`not`|`not`  |
|`lt` |`lt`   |
|`le` |`le`   |
|`gt` |`gt`   |
|`ge` |`ge`   |
|`eq` |`eq`   |

The difference between Bril and AArch64 is the addressing model.
AArch64 does the operation directly on the register data.
Bril does the operation on variables on stack, 
     which should be accessed by memory operations.

For example, `c:int = add a b`

1. load data `a` to `x8` by `ldr`
2. load data `b` to `x9` by `ldr`
3. `add x8, x8, x9`
4. store `x9` back to the space for `c`

## Other Instruction

* `const`: store value to stacok by `str`
* `id`: load the value by `ldr` and store to a different location by `str`
* `br`: load the register value to `x8`, then
    1. `cbz x8, label2`
    2. `b   lable1`
* `jmp`: `b label`
* `ret`: branch to the return point of current function
* `print`: calling the `printf` in C. 
There is a small generated function called `printbool` 
to print `true` or `false` string.

## Experiments
