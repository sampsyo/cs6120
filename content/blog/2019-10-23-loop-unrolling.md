+++
title = "Loop Unrolling Optimization"
extra.author = "Sameer Lal"
extra.email = "s j l 3 2 8@cornell.edu"
extra.bio = """
[Sameer Lal](https://github.com/sameerlal) is a Masters of Engineering student in the Computer Science. He studied ECE and Math in his undergrad.  He is interested in probability-related things.
"""
extra.latex = true
+++



## Goal
The goal of this project is to implement the loop unswitching optimization.  Loop unswitching involves detecting conditional expressions inside of loops whose condition is independent of the loop's body and then moving the condition outside of the loop with the replicated loop's body inside each branch of the conditional.  In other words, consider the following snippet of code:
```
bool b = false
x,y,z = 0
for _ in range(100):
    if b:
        x <= x + 1
    else:
        y <= y + 1
        z <= z + 1
```
Now since b is nonchanging inside of the loop's body, we can "unswitch" this code to be the following:
```
bool b = false
x,y,z = 0
if b:
    for _ in range(100):
        x <= x + 1
else:
    for _ in range(100):
        y <= y + 1
        z <= z + 1
```
Though code size has effectively doubled, we prevent the need to check the conditional statement while inside of the loop.  This leads to less branching and allows for optimizations within each conditional branch.

In this project, we implement loop unswitching for Bril.
## Design and Implementation
This optimization primarily involves loops, so the first step is to specify a contract for how loops are to be represented.  Note that other representations are allowed, but will need to be preprocessed into the following format before the loop unswitching optimization occurs:

```
for( int c = 0; c < max; c += i):
  x = x + x;
z = 11111
print(x)
print(z)
```
is represented as:
```
main:
  i: int = const 1;
  c: int = const 0
  max: int = const 10;
loopstart:
  b: bool = le c max;
  c: int = add c i;
  br b loopbody exitbranch;
loopbody:
  x: int = add x x;
  jmp loopstart;
exitbranch:
  z: int = const 11111;
  print x;
  print z;
}
```
Most importantly, logic containing whether to enter the loop is encoded in the block "loopstart."  From here, the program either branches to the loop's body which can consist of any number of blocks (here it is "loopbody") or it exits the loop and continues through the rest of the program (here, "exitbranch").  

There is also a standing assumption that we are working with Bril programs that have been transformed into SSA form.  Though this implementation does not require SSA form, it may not optimize conditionals that would otherwise be apparent in SSA form.
### Design

##### Loop Detection
We detect loops by producing the connected flow graph (CFG) for the program and then searching for a backedge whose tail is dominated by its head.  
```
back_edges = []
    for e in cfg_edges:
        head, tail = e
        if tail in dom_dic[head]:
            back_edges.append(e)
```
This creates a cycle.  Next, we find all nodes in between these two nodes by populating a stack with predecessors, until we have reached the beginning of the loop.

##### Deciding unswitchability
Now, we need to decide if this loop is unswitchable.  Recall that in order to unswitch loops, we need to ensure that the condition is independent of the loop's body during execution.  That is, the conditional statement we are unrolling cannot be modified in the loop.  This allows us to write deterministic code.

To implement this,  we adopt the following notation[1].  Let $$v_s$$ denote the set of variables defined by statements, and let $$v_a$$ denote the set of variables defined by arguments.  Now let $$V_b = v_a \cup v_s$$ be the union of the two for a block $$b$$, and let $$V_L = \cup_i V_{b_i}$$ denote the set of variables entirely in the loop.  Now suppose we have a branching statement on condition $$t$$:
```
br t if then;
```
Now if $$v_t \not \in V_L$$, we can unroll this loop.  In the case of multiple conditional statements that can be unrolled, as we traditionally do in literature, we pick one uniformly at random.

##### Implementing Unrolling
Once we have selected a subset of nodes in the CFG to be unrolled, we need to actually reorder the blocks.  At a high level, we implement the following reordering:


< DIAGRAM >

In the above diagram, we have the following:
* Before Loop Code:  This block represents all code before the start of the for loop
* For loop logic:  This consists of logic involving whether or not to enter the for loop's body.  Usually, this encodes code such as: ```for(int i=0; i<n; ++i). ```
* Loop Body (1):  This contains the entire loop body up until conditional t.  In particular, it can consist of many blocks, branches, conditionals, and nonconditional jumps.
* Conditional t:  This block consists of exactly one line of instruction which is in the form ``` br b if else``` where ```b``` is the branching boolean that is independent of the loop's body. 
* If Body: This block contains the contents of the ```if``` branch if ```b``` is true.
* Else Body:  This block contains the contents of the ```else``` branch if ```b``` is false.
* Loop body (2):  This contains the entire loop body following the conditional t.
* End of Program: This block contains all code after the loop.  In particular, it may contain additional loops with conditionals, that we are not optimizing.

To implement unrolling, we want to move the ```Conditional t``` block outside the for loop, create branches for each destination (in Bril, we are limited to two branches), and replicate the contents inside of the loop.  We wish to do surgury in such a way to only to disrupt nodes involving the loop, leaving the rest of the CFG intact.  A high level control flow is as follows for post-unrolling operation:

<< DIAGRAM >> 
In particular, we have the following blocks:
* Before loop code: This is the same block as before and contains the contents of the program before we enter the loop
* Conditional t:  This block contains one instruction, namely the branching instruction that involves the independent boolean.  Based on the value of the boolean, it connects to either the "if" or "else" blocks, each block containing its own for loop.
* If for loop logic:  This block is a replica of the ```for loop logic``` block in the previous CFG, except it branches to two newly created blocks.  If the program decides to enter the loop, we branch to ```If loop body``` and if it decides to exit the loop, it branches to bypass.  
* Else for loop logic:  This block is identical to the previous block, except it branches to either ```Else loop body``` or ```bypass``` depending on the loop invariant. 
* If loop body:  This contains all code in the ``if`` branch of the original CFG.  In particular, this block contains code that is ```Loop body (1)``` $$\cup$$ ```If body``` $$\cup$$ ```Loop body (2)```.  This block then automatically branches to a newly created block, ```jmp loop```.
* Else loop body: Similarly, this block contains code in the ```else``` branch of the original CFG:  ```Loop body (1)``` $$\cup$$ ```Else body``` $$\cup$$ ```Loop body (2)```.  This block then automatically branches to a newly created block (separate from the previous one), ```jmp loop```.
* jmp loop:  There are two of these blocks, and each acts as a proxy that feeds back into the loop logic.  Essentially, we delegate logic involving entering the loop through this block.  Futher optimizations can make use of this dummy block, though in this implementation, it contains exactly one jmp instruction.
* Bypass:  Both bypass blocks also delegate the program flow to the end of program block, essentially exiting the loop.
* End of program:  This block is identical to the ```End of Program``` block in the original CFG.

### Implementation
Implementing unrolling requires favoring generality over specificity.  




## Implementation Difficulties



## Evaluation and results



By and large, we have implemented the checker satisfying all of our defined behaviors. But we don't know if that's exhaustive for all possible errors (not necessarily type errors). We would be very happy if someone comes up with more cases and reaches out to us by mail or GitHub issues.