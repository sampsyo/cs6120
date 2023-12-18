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

# Summary
 [John Rubio](https://github.com/JohnDRubio) and I wrote a Python program that converts Bril to runnable RISC-V assembly for our final project.


- [Bril Insn Classes + convert to RISC-V functions](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/BrilInsns)
- [RISC Insn Classes](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/RVIRInsns)
- [Trivial Register Allocation Logic](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/TrivialRegAlloc)
- Calling Convention Logic
  - [Prologue Inserter](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/util/prologue.py)
  - [Epilogue Inserter](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/util/epilogue.py)
  - [Lowering function call](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/BrilInsns/BrilFunctionCallInsn.py)

# The goal

Bril (TODO: ADD LINK) is a user-friendly, educational intermediate language. Bril programs have typically been run using the Bril interpreter (TODO: ADD LINK). Compiling Bril programs to assembly code that can run on real hardware would allow for more accurate measurements of the impacts of compiler optimizations on Bril programs in terms of execution time or clock cycles. Thus, the goal of [this project](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend) was to write a RISC-V backend. That is, to write a program that lowers the core (TODO: ADD LINK) subset of Bril to TinyRV32IM (TODO: ADD LINK). The objective was to ensure semantic equivalence between the source program and the generated RISC-V code by running it on a RISC-V emulator. At the outset of this project, one of the stretch goals was to use Crocus (TODO: ADD LINK) to verify the correctness of the Bril-to-RISC-V lowering rules. Another stretch goal was to perform a program analysis step that would aid in instruction selection, allowing the lowering phase to take place in an M-to-N fashion as opposed to the more trivial 1-to-N approach. The authors regret to inform you that these stretch goals were not completed during the semester, however, the primary goal was achieved. The primary goal was to generate semantically equivalent RISC-V assembly code from a Bril source program using a dead simple approach: 1-to-N instruction selection, trivial register allocation, and correct calling conventions.

## Bril Instruction classes

The first step in this project included writing classes for each Bril instruction. For code reuse, we organized these classes in a hierarchy that would 
allow for similarities in instructions to be in common classes using inheritance. The main idea for this is if we needed to tweak these common aspects in 
the instruction classes, the change would be in one place rather than having to keep track of all the different files to update. The other reason why we dealt 
with objects rather than bril text or bril JSON, was because it is easier to keep track and update operands in an object format.

The class hierarchy we came up with was the following:

<img width="1689" alt="Screenshot 2023-12-11 at 6 46 26 PM" src="https://github.com/20ashah/cs6120/assets/33373825/5b165c72-39bf-44d0-93b5-2dc37a265bb9">

## Parse Bril to Objects

Once we had classes for each Bril instruction, now we needed to actually create these objects from a Bril program. We performed the following steps for each 
function in the Bril program:

1. Form basic blocks
      - Not entirely necessary, but in the event that we want to perform optimizations later on, it would have been helpful for these bril instructions to be in basic blocks
2. Insert missing labels on basic blocks
      - For basic blocks that didn’t have a label, a unique one was assigned - mainly for the purpose of creating a more complete CFG.
3. Mangle variable names by prepending each variable name with an underscore
      - In the event that the user-defined a variable x1, we did not want this to be confused with the actual register in RISC-V during register allocation, so we added an
        underscore before every variable in the Bril program.
4. Parse instructions in Bril JSON
      - The main part of this step includes going through each Bril instruction (which is a	JSON object), picking out certain parts of the instruction such as ‘op’, and
        identifying what kind of instruction it is. Once this was determined, we created an instance of the appropriate Bril class. By the end of this process, we have a
        list of Bril object instructions for each function in the Bril program.

## RISC-V Instruction classes

There was less of a hierarchy in our RISC-V instruction classes since there were not that many different types of instructions like there were in Bril. For these classes, we had a general RISC-V instruction class that had method stubs such as emitting assembly, getting abstract registers, getting read/write registers (explained in detail below), and converting registers from abstract to real that each of the RISC-V classes would implement.

## Convert Bril to RISC-V IR Objects (Abstract Assembly)

Now that we have a list of Bril object instructions and a hierarchy of RISC-V classes, we ultimately want a list of RISC-V object instructions that is semantically equivalent 
to the list of Bril object instructions. To implement this, we had each Bril instruction class implement a function to convert itself into a series of RISC-V objects. 
As specified before, we implemented this as a 1-N approach, where each Bril instruction corresponds to N RISC-V instructions. This is less efficient than a N-N approach, 
where we try and look for Bril instructions we can combine for optimization, but the 1-N approach was the first step in generating working RISC-V assembly, which was our first goal.

An important note here is that in the first pass, we implemented this for every Bril instruction object except for function calls. The reason for this is that we wanted to save 
the calling convention pass of lowering until the end, even after register allocation. We anticipated that this part would be the hardest, and so we wanted to get RISC-V assembly 
without function calls working first before we added that whole layer of complexity. This worked out quite well, since we could test the correctness of the assembly we were generating
without function calls early on, without having implemented lowering for function calls.

Below is a brief description of the conversions from Bril instructions to RISC-V instructions. We filled out this table prior to actually coding these functions to make sure 
that logically, our conversions created RISC-V instructions that were semantically equivalent to the Bril. Note that this is abstract assembly, so no actual RISC-V registers are used.


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
| print x                         | nop                               |
| jmp .label                      | jal x0, .label                    |
| br cond .label1 .label2         | addi tmp, x0, 1 <br> beq cond, tmp, .label1 <br> jal x0, .label2 <br> .label1: <br> ... <br> jal x0 .exit <br> .label2: <br> ... <br> .exit: |
| ret x                           | addi a0, x, 0 <br> jalr x0, x1, 0 |
| ret                             | jalr x0, x1, 0                    |
| x: int = id y                   | addi x, y, 0                      |
 
                                  
Note: An important note about the above chart that required some extra implementation had to deal with the case when the RISC-V instructions needed to add in temps and labels 
to match the behavior of the Bril instructions. An edge case here is that these labels and temp variables need to be generated fresh each time for semantic equivalence, so 
keeping track of this was a key part of the converter.

## Trivial Register Allocation

At this stage of compilation, we have lowered a Bril program to a list of instructions, all of which are RISC-V class objects with abstract registers, with the exception of 
function calls, which still remain Bril objects (We saved dealing with calling conventions until the end). In this stage of lowering, we again ignore the function calls, 
but with all of the RISC-V objects, we get rid of the abstract registers and replace them with actual RISC-V registers using trivial register allocation. Before discussing the 
details of how this was implemented, we will first briefly describe how trivial register allocation works, in a non-architecture-specific way.

### Description of trivial register allocation

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
two separate offsets in each of these functions (for obvious reasons). By the end of this, for each function in our bril program, we have an object that has mapped each unique abstract variable to a unique stack location.

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

We found it surprisingly easy to convert a Bril program to abstract assembly and even perform trivial register allocation. Because Bril was already a flat set of instructions 
with instructions very similar to RISC-V, these initial passes to create RISC-V instructions were fairly simple. Trivial register allocation was slightly more complicated, 
but still was easily implemented with a mapper to assign and keep track of stack offsets. The hardest part to get right was definitely lowering function calls via RISC-V calling conventions. 

A meta-problem we ran into was finding an environment that runs basic, 32-bit RISC-V assembly code. Due to a lack of foresight, we underestimated the amount of work required to get such an environment set up. In any case, we ended up settling on a few RISC-V interpreters that display the architectural state at the end of each program execution. Note that we did not add support for `print` instructions. Using an admittedly error-prone and somewhat monotonous procedure, we lowered Bril [programs](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/test) to RISC-V using our system. Next, we wrote C++ programs that were semantically equivalent to the same Bril programs and obtained the compiled RV32 code using [Compiler Explorer](https://godbolt.org/). Lastly, we ran both our RISC-V program and the godbolt RISC-V program on RISC-V interpreters and compared the end architectural state of each program. On the programs we tested, our RISC-V programs yielded identical architectural states to the Compiler Explorer programs, save quite a few more items on the stack in our case since we used trivial register allocation. Obviously, there are some issues with this approach - it does not provide a high degree of coverage and it does not provide any details on performance. We concede these two points. This approach does show that, at the very least, our lowering system can produce RISC-V programs that are semantically equivalent to non-trivial Bril programs. We would've liked to explore performance speedups in further detail but ultimately information on the performance of the JavaScript runtime and that of compiled RISC-V programs is not difficult to find.

# Were we successful?

Our original goal was to implement a RISC-V lowering system, implement Bril-to-RISC-V lowering rules using Cranelift's ISLE DSL, and to use Crocus to verify the semantic equivalence of our lowering system against that of Cranelift. We learned that we were a bit overambitious in these goals and that implementing the Bril-to-RISC-V lowering rules in Cranelift's ISLE DSL required enough overhead to be its own project. So we were unsuccessful in achieving goals (2) and (3), but we were successful in achieving goal (1).

Our goal to generate runnable RISC-V assembly from a Bril program was successful. We wanted to prioritize getting a working version of this assembly instead of fancier optimizations, which was achieved in this project. In terms of how we went about testing, we took a unit testing style approach, of testing individual modules (in this case Bril instructions) to make sure that the lowered RISC-V instructions made sense and were semantically equivalent. After identifying that this lowering worked on instructions in isolation, we tested these instructions in various combinations.

Since we implemented trivial register allocation before function call lowering, we first tested our compiler thoroughly with [test cases](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/test/trivial-reg-alloc) that encompassed every Bril instruction minus function calls / arguments. Once we verified that this worked, we moved on to testing with function calls once the calling conventions part was implemented. The main thing we tested was that we could execute this RISC-V assembly and it was semantically equivalent to the Bril program.

Just to highlight our results in some way, [here](https://github.com/JohnDRubio/CS_6120_Advanced_Compilers/tree/main/rv32_backend/asm) are some examples that illustrate outputted RISC-V assembly that we generated from Bril programs.

