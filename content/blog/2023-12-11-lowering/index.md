+++
title = "Bril to RISC-V Lowering System"
[extra]
bio = """
John Rubio is a 2nd year MSCS student interested in hardware and compilers. In his free time, he trains Brazilian Jiu-Jitsu. 
Arjun Shah is a senior undergraduate majoring in CS. Arjun is a Java enthusiast who is interested in [ultra-large-scale software systems](https://en.wikipedia.org/wiki/Ultra-large-scale_systems). In his free time, Arjun trains for competitive weight-lifting.
"""
latex = true
[[extra.authors]]
name = "John Rubio"
[[extra.authors]]
name = "Arjun Shah"
+++

TODO: Move these links

- [Trivial Register Allocation Logic](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/TrivialRegAlloc)
- Calling Convention Logic
  - [Prologue Inserter](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/util/prologue.py)
  - [Epilogue Inserter](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/util/epilogue.py)
  - [Lowering function call](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/BrilInsns/BrilFunctionCallInsn.py)

# Summary
Bril (TODO: ADD LINK) is a user-friendly, educational intermediate language. Bril programs have typically been run using the Bril interpreter (TODO: ADD LINK). Compiling Bril programs to assembly code that can run on real hardware would allow for more accurate measurements of the impacts of compiler optimizations on Bril programs in terms of execution time or clock cycles. Thus, the goal of [this project](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend) was to write a RISC-V backend. That is, to write a program that lowers the [core subset of Bril](https://capra.cs.cornell.edu/bril/lang/core.html) to TinyRV32IM (TODO: ADD LINK), a subset of RV32IM (TODO: ADD LINK). The objective was to ensure semantic equivalence between the source program and the generated RISC-V code by running it on a RISC-V emulator. At the outset of this project, one of the stretch goals was to use Crocus (TODO: ADD LINK) to verify the correctness of the Bril-to-RISC-V lowering rules. Another stretch goal was to perform a program analysis step that would aid in instruction selection, allowing the lowering phase to take place in an M-to-N fashion as opposed to the more trivial 1-to-N approach. The authors regret to inform you that these stretch goals were not completed during the semester, however, the primary goal was achieved. The primary goal was to generate semantically equivalent RISC-V assembly code from a Bril source program using a dead simple approach: 1-to-N instruction selection, trivial register allocation, and correct calling conventions.

## Representing Bril Programs

The first stage in the lowering pipeline is a preprocessing step. Source Bril programs are provided as input in JSON format. The program is parsed and, for each function, each Bril instruction is translated to one [__BrilInsn__](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/BrilInsns) object. Each BrilInsn is an instance of a subclass of the BrilInsn class hierarchy as depicted in __Figure 1__ below. The reasoning behind the structure of the BrilInsn class hierarchy lies in the fact that [most Bril instructions have a similar format](https://capra.cs.cornell.edu/bril/tools/text.html). This observation motivated a more conventional, Object-Oriented (OO) approach since the common Bril instruction formats could be implemented as parent classes and the small number of deviations from these common formats could be captured in the form of child classes. The BrilInsn class hierarchy lends itself to exploiting some of the main benefits of OO, namely minimal changes and maximal code reuse. For example, a [value operation](https://capra.cs.cornell.edu/bril/lang/syntax.html#:~:text=string%3E%22%2C%20...%5D%3F%2C%0A%20%20%22labels%22%3A%20%5B%22%3Cstring%3E%22%2C%20...%5D%3F%20%7D-,A%20Value%20Operation,-is%20an%20instruction) is a general type of Bril instruction that takes arguments, performs some computations, and produces a value. Several types of Bril instructions fall under the umbrella of value operations, namely arithmetic and logical operation instructions, ID assignments, and function calls. This inherent hierarchical structure is a perfect opportunity for subclassing. The one attribute each of these Bril instruction types have in common is a destination field which is why __Figure 1__ shows the *BrilValueOperation* abstract class with a single `dest` field. The specifics of the computations that arithmetic & logical instructions, ID assignments, and function calls differ enough to justify each of these instruction types being their own subclass of the *BrilValueOperation* class. Using an OO approach allowed us to minimize the amount of time dedicated to common scaffolding among classes and focus more on implementation details specific to a class.

<img width="1689" alt="UML Diagram" src="BrilInsn_Class_Hierarchy2.jpeg">

 __Figure 1__

## Representing RISC-V Programs

As mentioned above, a __1-to-N__ instruction selection approach was used for lowering. Thus, for each Bril instruction, one or more relatively simple RISC-V instructions would be used. The subset of RISC-V being used is TinyRV32IM which only consists of about 30 non-privileged RV32IM instructions. After some consideration, we felt the most straightforward way to group these instructions was the following:

- Register-Register Arithmetic Instructions
- Register-Immediate Arithmetic Instructions
- Memory Instructions
- Unconditional Jump Instructions
- Conditional Jump Instructions

Each of these groups corresponds to a class in the RISC-V Intermediate Representation (RVIR) [implementation](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/RVIRInsns). RISC-V instructions are fundamentally simple, thus there was no need for a class hierarchy other than each RVIR instruction inheriting from the [RVIRInsn abstract class](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/blob/main/rv32_backend/RVIRInsns/RVIRInsn.py).

## Progressive Lowering

With the proper infrastructure in place, it is possible to perform the first of two lowering steps. As shown in __Figure 1__, each BrilInsn instance implements a *conv_riscvir* method. As the name implies, this method converts each BrilInsn instance to one or more RVIRInsn instances, thus implementing the 1-N instruction selection design. See __Table 1__ below for translation details. Each RVIRInsn instance corresponds to a single RISC-V IR instruction. For each function in the source Bril program, this pass returns a list of RVIRInsn instances representing a semantically equivalent RVIR function. It is worth noting that the only difference between RIVR instructions and true RISC-V instructions is that RVIR instructions do not use RISC-V registers (TODO: ADD LINK). To lower to true RISC-V from RVIR, a register allocation pass is required.

TODO: Clean up table

| Bril                            | RISC-V Abstract Asm                |
| ------------------------------- | --------------------------------- |
| x: int = const 1                | addi x, x0, 1                     |
| x: int = add y z                | add x, y, z                       |
| x: int = mul y z                | mul x, y, z                       |
| x: int = sub y z                | sub x, y, z                       |
| x: int = div y z                | div x, y, z                       |
| x: bool = eq y z                | beq y, z, .eq <br> addi x, x0, 0 <br> jal x0 .exit_cond <br> .eq: <br> addi x, x0, 1 <br> .exit_cond: |
| x: bool = lt y z                | blt y, z, .lt <br> addi x, x0, 0 <br> jal x0 .exit_cond <br> .lt: <br> addi x, x0, 1 <br> .exit_cond: |
| x: bool = gt y z                | blt z, y, .gt <br> addi x, x0, 0 <br> jal x0 .exit_cond <br> .gt: <br> addi x, x0, 1 <br> .exit_cond: |
| x: bool = le y z                | bge z, y, .le <br> addi x, x0, 0 <br> jal x0 .exit_cond <br> .le: <br> addi x, x0, 1 <br> .exit_cond: |
| x: bool = ge y z                | bge y, z, .ge <br> addi x, x0, 0 <br> jal x0 .exit_cond <br> .ge: <br> addi x, x0, 1 <br> .exit_cond: |
| x: bool = not y                 | xori x, y, 1                      |
| x: bool = and y z               | and x, y, z                       |
| x: bool = or y z                | or x, y, z                        |
| jmp .label                      | jal x0, .label                    |
| br cond .label1 .label2         | addi tmp, x0, 1 <br> beq cond, tmp, .label1 <br> jal x0, .label2 <br> .label1: <br> ... <br> jal x0 .exit <br> .label2: <br> ... <br> .exit: |
| ret x                           | addi a0, x, 0 <br> jalr x0, x1, 0 |
| ret                             | jalr x0, x1, 0                    |
| x: int = id y                   | addi x, y, 0                      |

 __Table 1__

## Trivial Register Allocation

At this stage of compilation, we have lowered a Bril program to a list of instructions, all of which are RISC-V class objects with abstract registers, with the exception of 
function calls, which still remain Bril objects (We saved dealing with calling conventions until the end). In this stage of lowering, we again ignore the function calls, 
but with all of the RISC-V objects, we get rid of the abstract registers and replace them with actual RISC-V registers using trivial register allocation. Before discussing the 
details of how this was implemented, we will first briefly describe how trivial register allocation works, in a non-architecture-specific way.

### Overview

The first step is to select 3 caller-saved registers in the given assembly language. In most assembly languages, instructions deal with at most only 2 registers at a time, 
so we’ll never need to have more than these registers. Next, the main part of trivial register allocation involves adding instructions before each abstract assembly 
instruction to shuttle variables off of the stack into these registers, and instructions after to shuttle variables back onto the stack. This design of allocation has these 
hardware registers simply being shuttle temps, moving data on and off of the stack, and never really storing important variable data after that function has passed. Obviously, this 
implementation is not very efficient, since this requires loads and stores for every abstract assembly instruction, but it is the simplest implementation to create working assembly code.

### Implementation

The way this was implemented in our case was first creating a class that dealt with keeping track of stack offsets for each variable. What this class did was go through our list
of RISC-V instruction objects and for each one, pull out all of the abstract temps - variables that were not real RISC-V registers and would need to be included in trivial register
allocation. For each of these temps, this mapping class would assign it a unique offset in increments of 8 (starting from -8). 

This was implemented as a dictionary in Python. So, if we come across a variable that is already in this dictionary, this means that we have seen it before and had already assigned it
an offset, so we move on. Note that this happens on a per-function basis, so if we had two separate functions and they both contain a variable 'a', this variable is getting assigned 
two separate offsets in each of these functions (for obvious reasons). By the end of this, for each function in our Bril program, we have an object that has mapped each unique abstract variable to a unique stack location.

The next step is to actually perform the trivial register allocation for each instruction in our function. As described above, this involves adding additional instructions before and after each instruction to load the appropriate temps off the stack, perform the operation, and then store the temps back onto the stack. The important property of when to add instructions before or after an instruction X is the following:

- For each temporary variable that X reads from, we must include loads before the instruction to load these variables from their stack locations  
- For the temporary variable that X writes to, we must include a store after the instruction to store this variable to its appropriate stack location

These conditions led us to implement functions for each RISC-V instruction object - one to return the list of abstract temps that the instruction reads from, and one to return the list of abstract temps that the instruction writes to. Knowing these for each instruction would allow us to determine where to place surrounding instructions to shuttle temps on and off the stack.

Now, we iterate through each RISC-V instruction and get the list of abstract temps that it reads from and writes to. For each of the abstract temps that the instruction reads from, we use our mapper to look through the dictionary and find the stack offsets for each. Before this instruction, we add loads from these stack offsets into our RISC-V registers. Then we perform the operation in the instruction, being sure to replace the abstract temps that are read from with the RISC-V registers that we just loaded into. Now that the abstract temps in the instruction that we read from are taken care of, we move on to writes. We look at the abstract temp we write to, and replace that in the instruction with the next available RISC-V register we haven't used for the reads yet. We then find using the mapper the stack location of this destination variable and we add an instruction after the operation that performs a store from the destination RISC-V register into the appropriate stack offset the mapper gave us.

Doing this for each RISC-V instruction gives us executable assembly that is complete without any abstract temps besides dealing with function calls/arguments and calling conventions. 

## Calling Conventions

By far, the biggest implementation obstacle was implementing the RISC-V calling conventions. We based our implementation off of the RISC-V calling conventions taught in Cornell's CS 3410: Computer System Organization and Programming. To oversimplify four lectures' worth of material, the essence of RISC-V calling conventions can be distilled into three main phases:

1. **Prologue (Function Entry Setup):** Before entering the function body, a set of preparatory actions, known as the prologue, take place. These include the creation of the stack frame. The size of the stack frame usually needs to be known ahead of time, considering the space needed to accommodate the return address, frame pointer, any overflow arguments, and local variables that must be pushed onto the stack. Additionally, all callee-save registers that will be utilized during the function execution should be pushed onto the stack at this stage.

2. **Function Call Instruction Preparation:** Prior to a function call instruction, certain steps must be taken to ensure the proper transfer of control and data. This involves placing function arguments in designated registers, following the argument-passing registers convention. For additional arguments or those that don't fit into the designated registers, space on the stack is allocated to hold these values. The function call instruction is then executed, initiating the transfer of control to the callee.

3. **Epilogue (Clean-up):** The third key piece of this calling convention puzzle is the clean-up step, or the epilogue. After the instructions in the function body have finished executing, the epilogue is responsible for restoring the stack to its original state and releasing any resources allocated during the prologue. This includes popping the stack frame, restoring the values of callee-save registers, and ensuring a smooth return to the calling function.

# What were the hardest parts to get right?

While implementing the initial translations from Bril to RVIR, some subtle details were tricky to get right. For example, conditional assignments such as `x: bool = lt y z` were implemented using a sequence of RVIR instructions that included a branch, two labels, and a jump. A set of fresh labels needed to be generated for each conditional assignment statement, otherwise the program could contain two or more identical labels and the compiler would crash.

We found it surprisingly easy to convert a Bril program to abstract assembly and even perform trivial register allocation. Because Bril was already a flat set of instructions 
with instructions very similar to RISC-V, these initial passes to create RISC-V instructions were fairly simple. Trivial register allocation was slightly more complicated, 
but still was easily implemented with a mapper to assign and keep track of stack offsets. The hardest part to get right was definitely lowering function calls via RISC-V calling conventions. 

A meta-problem we ran into was finding an environment that runs basic, 32-bit RISC-V assembly code. Due to a lack of foresight, we underestimated the amount of work required to get such an environment set up. In any case, we ended up settling on a few RISC-V interpreters that display the architectural state at the end of each program execution. Note that we did not add support for `print` instructions. Using an admittedly error-prone and somewhat monotonous procedure, we lowered Bril [programs](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/test) to RISC-V using our system. Next, we wrote C++ programs that were semantically equivalent to the same Bril programs and obtained the compiled RV32 code using [Compiler Explorer](https://godbolt.org/). Lastly, we ran both our RISC-V program and the Compiler Explorer RISC-V program on RISC-V interpreters and compared the end architectural state of each program. On the programs we tested, our RISC-V programs yielded identical architectural states to the Compiler Explorer programs, save quite a few more items on the stack in our case since we used trivial register allocation. Obviously, there are some issues with this approach - it does not provide a high degree of coverage and it does not provide any details on performance. We concede these two points. This approach does show that, at the very least, our lowering system can produce RISC-V programs that are semantically equivalent to non-trivial Bril programs. We would've liked to explore performance speedups in further detail but ultimately information on the performance of the JavaScript runtime and that of compiled RISC-V programs is not difficult to find.

# Were we successful?

Our original goal was to implement a RISC-V lowering system, implement Bril-to-RISC-V lowering rules using Cranelift's ISLE DSL, and to use Crocus to verify the semantic equivalence of our lowering system against that of Cranelift. We learned that we were a bit overambitious in these goals and that implementing the Bril-to-RISC-V lowering rules in Cranelift's ISLE DSL required enough overhead to be its own project. So we were unsuccessful in achieving goals (2) and (3), but we were successful in achieving goal (1).

Our goal to generate runnable RISC-V assembly from a Bril program was successful. We wanted to prioritize getting a working version of this assembly instead of fancier optimizations, which was achieved in this project. In terms of how we went about testing, we took a unit testing style approach, of testing individual modules (in this case Bril instructions) to make sure that the lowered RISC-V instructions made sense and were semantically equivalent. After identifying that this lowering worked on instructions in isolation, we tested these instructions in various combinations.

Since we implemented trivial register allocation before function call lowering, we first tested our compiler thoroughly with [test cases](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/test/trivial-reg-alloc) that encompassed every Bril instruction minus function calls / arguments. Once we verified that this worked, we moved on to testing with function calls once the calling conventions part was implemented. The main thing we tested was that we could execute this RISC-V assembly and it was semantically equivalent to the Bril program.

Just to highlight our results in some way, [here](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/asm) are some examples that illustrate outputted RISC-V assembly that we generated from Bril programs.

