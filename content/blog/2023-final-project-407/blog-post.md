# Summary
TODO

# What was the goal?

The goal for this project was to write the backend of a compiler that a bril program and lowered it to RISC V assembly - which ultimately 
we could run on a RISC V emulator to verify that the outputted RISC V program behaved the same as the bril program. The goal was just 
to get a working version of RISC V assembly. We thought if we had time after doing this, it might be nice to make the assembly more efficient, 
but our first goal was a working version. We wanted to implement this lowering as a 1-N approach, where each bril instruction would be mapped 
to a certain number of RISC V instructions to get an abstract assembly. And then simply implementing calling conventions and trivial register allocation. 
Each of these parts are described in detail below.

# What did you do? (Include both the design and the implementation.)

## Bril Instruction classes

The first step in this project included writing classes for each Bril instruction. For code reuse, we organized these classes in a hierarchy that would 
allow for similarities in instructions to be in common classes using inheritance. The main idea for this is if we needed to tweak these common aspects in 
the instruction classes, the change would be in one place rather than having to keep track of all the different files to update. The other reason why we dealt 
with objects rather than bril text or bril json, was because it is easier to keep track and update operands in an object format.

The class hierarchy we came up with was the following:

<img width="1689" alt="Screenshot 2023-12-11 at 6 46 26 PM" src="https://github.com/20ashah/cs6120/assets/33373825/5b165c72-39bf-44d0-93b5-2dc37a265bb9">

TODO: Explain class hierarchy

## Parse Bril to Objects

Once we had classes for each bril instruction, now we needed to actually create these objects from a Bril program. We performed the following steps for each 
function in the Bril program:

1. Form basic blocks
      - Not entirely necessary, but in the event that we want to perform optimizations later on, it would have been helpful for these bril instructions to be in basic blocks
2. Insert missing labels on basic blocks
      - For basic blocks that didn’t have a label, a unique one was assigned - mainly for the purpose of creating a more complete CFG.
3. Add underscore before every variable name
      - In the event that the user defined a variable x1, we did not want this to be confused with the actual register in RISC V during register allocation, so we added an
        underscore before every variable in the Bril program.
4. Parse instructions in Bril json
      - The main part in this step includes going through each Bril instruction (which is a	json object), picking out certain parts of the instruction such as ‘op’, and
        identifying what kind of instruction it is. Once this was determined, we created an instance of the appropriate Bril class. By the end of this process, we have a
        list of Bril object instructions for each function in the bril program.

## RISC V Instruction classes

TODO: add screenshot of hierarchy
TODO: Explain class hierarchy

## Convert Bril to RISC V Objects (Abstract Assembly)

Now that we have a list of Bril object instructions, and a hierarchy of RISC V classes, we ultimately want a list of RISC V object instructions that is semantically equivalent 
to the list of Bril object instructions. To implement this, we had each Bril instruction class implement a function to convert itself into a series of RISC V objects. 
As specified before, we implemented this as a 1-N approach, where each Bril instruction corresponds to N RISC V instructions. This is less efficient than a N-N approach, 
where we try and look for Bril instructions we can combine for optimization, but the 1-N approach was the first step in generating working RISC V assembly, which was our first goal.

An important note here is that in the first pass, we implemented this for every Bril instruction object except for function calls. The reason for this is because we wanted to save 
the calling convention pass of lowering until the end, even after register allocation. We anticipated that this part would be the hardest, and so we wanted to get RISC V assembly 
without function calls working first before we added that whole layer of complexity. This worked out quite well, since we could test the correctness of the assembly we were generating
without function calls early on, without having implemented lowering for function calls.

Below is a brief description of the conversions from Bril instructions to RISC V instructions. We filled out this table prior to actually coding these functions to make sure 
that logically, our conversions created RISC V instructions that were semantically equivalent to the Bril. Note that this is abstract assembly, so no actual RISC V registers are used.

TODO: add table from google doc

Note: An important note about the above chart that required some extra implementation had to deal with the case when the RISC V instructions needed to add in temps and labels 
to match the behavior of the Bril instructions. An edge case here is that these labels and temp variables need to be generated fresh each time for semantic equivalence, and so 
keeping track of this was a key part of the converter.

## Trivial Register Allocation (without calls)

At this stage of compilation, we have lowered a Bril program to a list of instructions, all of which are RISC V class objects with abstract registers, with the exception of 
function calls, which still remain to be Bril objects (We saved dealing with calling conventions until the end). In this stage of lowering, we again ignore the function calls, 
but with all of the RISC V objects, we get rid of the abstract registers and replace them with actual RISC V registers using trivial register allocation. Before discussing the 
details of how this was implemented, we will first briefly describe how trivial register allocation works, in a non architecture specific way.

### Description of trivial register allocation

The first step is to select 3 caller saved registers in the given assembly language. In most assembly languages, instructions deal with at most only 2 registers at a time, 
and so we’ll never need to have more than these registers. Next, the main part of trivial register allocation involves adding instructions before each abstract assembly 
instruction to shuttle variables off of the stack into these registers, and instructions after to shuttle variables back onto the stack. This design of allocation has these 
hardware registers simply being shuttle temps, moving data on and off of the stack, never really storing important variable data after that function has passed. Obviously, this 
implementation is not very efficient, since this requires loads and stores for every abstract assembly instruction, but it is the simplest implementation to create working assembly code.

### How this was implemented

The way this was implemented in our case was first creating a class that dealt with keeping track of stack offsets for each variable.
TODO: finish

## Calling Conventions

TODO

# What were the hardest parts to get right?

We found it surprisingly easy to convert a Bril program to abstract assembly and even perform trivial register allocation. Because Bril was already a flat set of instructions 
with instructions very similar to RISC V, these initial passes to create RISC V instructions were fairly simple. Trivial register allocation was slightly more complicated, 
but still was easily implemented with a mapper to assign and keep track of stack offsets.

The hardest part to get right was definitely lowering function calls via RISC V calling conventions.


# Were you successful? (Report rigorously on your empirical evaluation.)

Yes - if we wanted to, we could say we hoped to finish a little earlier, so we could implement some optimizations, but we didn’t get to that point. But at least we got 
working RISC V assembly.

