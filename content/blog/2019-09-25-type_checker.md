+++
title = "Static Type Checking for Bril"
extra.author = "Yi Jing, Zhijing Li, Neil Adit"
extra.bio = """
[Zhijing Li](https://tissue3.github.io/) is a 2nd year Phd student at ECE department, Cornell. She is interested in compiler optimization and domain-specific languages. \n
[Yi Jiang](http://www.cs.cornell.edu/~yijiang/) is a 2nd year Phd student in CS dept, Cornell. She is interested in designing compilers for Processing-In-Memory related architectures and also general compiler optimization. \n
[Neil Adit](http://www.neiladit.com) is (guess what!) also a 2nd year PhD student at CSL, Cornell. He is interested in fun things in compiler research.

"""
extra.latex = true
+++



## Goal

The goal of the project was to add a static type checker to find type errors, multiple definitions of variables and undefined ones as well. We also ensure that the branch labels are valid and unique.



## Design and Implementation

### Design


To establish type checking rules we define a basic environment $\sigma$:

$$\sigma: \\var: int | bool\\label: strings | \#line$$

We start by defining each variable as either an integer or a boolean - the two valid types in Bril.

$$ \frac{}{<n,\sigma>\Downarrow_a int} \rightarrow \frac{}{<v,\sigma>\Downarrow_a var_{int}}\\ \frac{}{<true/false,\sigma>\Downarrow_b bool} \rightarrow \frac{}{<v,\sigma>\Downarrow_a var_{bool}}\\ $$


Then we define the arithmetic rules where $int$ is a constant integer, while $var_{int}$ indicates variable of integer type. We note the fact that *arithmetic operations* only take integer variables as input and produce an integer variable as output.

$$ \frac{<a_1,\sigma>\Downarrow_a var_{int}, <a_2,\sigma>\Downarrow_a var_{int}}{<a_1+a_2,\sigma>\Downarrow_a var_{int}}\\ \frac{<a_1,\sigma>\Downarrow_a var_{int}, <a_2,\sigma>\Downarrow_a var_{int}}{<a_1-a_2,\sigma>\Downarrow_a var_{int}}\\ \frac{<a_1,\sigma>\Downarrow_a var_{int}, <a_2,\sigma>\Downarrow_a var_{int}}{<a_1 * a_2,\sigma>\Downarrow_a var_{int}}\\ \frac{<a_1,\sigma>\Downarrow_a var_{int}, <a_2,\sigma>\Downarrow_a var_{int}}{<a_1 / a_2,\sigma>\Downarrow_a var_{int}} $$

Similarly, for *boolean operations*, we have the followings where  $bool$ is a constant boolean value, while $var_{bool}$ indicates variable of boolean type.

$$ \frac{<b_1,\sigma>\Downarrow_b var_{bool}}{<not\ b_1,\sigma>\Downarrow_b var_{bool}}\\ \frac{<b_1,\sigma>\Downarrow_b var_{bool}, <b_2,\sigma>\Downarrow_b var_{bool}}{<and/or\ b_1 b_2,\sigma>\Downarrow_b var_{bool}} $$

In *comparison operation* we note that integer variables are compared to give a boolean output:

$$ \frac{<a_1,\sigma>\Downarrow_a var_{int}, <a_2,\sigma>\Downarrow_a var_{int}}{<eq/lt/gt/le/ge\ a_1 a_2,\sigma>\Downarrow var_{bool}} $$

For *control flow operations* we need to make sure that the $label$ in the operation actually exists in the program and is unique. We can see this via the rule below where $\sigma[label]$ is all the labels.

$$ \frac{l_1 \in \sigma[label]}{jmp\ l_1,\sigma \Downarrow \sigma'} $$

For the branch condition we check that the inputs are boolean and valid labels and jump to a different environment denoted by $\sigma'$

$$ \frac{<cond,\sigma>\Downarrow_b bool, l_1\in \sigma[label], l_2\in \sigma[label]}{<br cond l_1 l_2, \sigma>\Downarrow \sigma'}\\ $$

There are some special operator: const, id, print, ret. We list the rules below. Though we are not very confident at formulation on this part, it should not affect the correctness of the our type checking program.

$$ \frac{<n,\sigma>\Downarrow int}{<var:int = const\ n ,\sigma>\Downarrow \sigma'}\\ \frac{a,\sigma \Downarrow_a var_{int}}{<id\ a,\sigma>\Downarrow \sigma'}\\ \frac{b,\sigma \Downarrow_b var_{bool}}{<id\ b,\sigma>\Downarrow \sigma'}\\ \frac{}{<ret,\sigma>\Downarrow End}\\ \frac{\forall v_i \in \sigma[var]}{<print\ v_1,v_2,\dots,v_i,\dots,v_n,\sigma> \Downarrow \sigma'} $$



### Implementation

Once we had the rules setup for all types of instructions, we check each instruction against them. We have a first pass in the algorithm which collects a list of labels and ensures that each label is unique. This also helps to check for control instructions if the label in the argument is valid.  In the second pass, we go over each instruction and check for various type errors - 

1. For const instructions, check if assigned value is of the same type as the destiantion type. We also check if the variable has already been assigned to a different type in that context, in which case reassignment would be an error.
2. For arithmetic and logical operations, we check if the two source operands are valid and are of the same type as the destination type.
3. For comparison operations, the source operands should be valid integer type variables and the destination should be a boolean.
4. For jump and branch instructions, making sure that label is valid and the condition argument is a boolean type suffices.



## Hardest parts during the implementation

The type checker implementation, though straightforward, had a few challenges. 

1. Type check rules had to be designed specifically for each instruction. While implementing these rules, we had to carefully partition all the cases and link them to appropriate arithmetic/boolean/control checks.
2. Keeping track of the line number and error message to return when encountered with an error.
3. Mantaning modularity to help future developments like checking expressions recursively.



## Evaluation and results

We wrote a comprehensive set of benchmaking codes to test our implementation for various types of rules defined in the design section above. 

The following table represents instructions that should throw up an error when compiled using the type checker system developed for Bril.


| Instruction |                 Type check rule for testing                  |  Pseudo Code snippet  |
| ----------- | :----------------------------------------------------------: | :------: |
| Add         | Adding an integer and a boolean type | ` int a; bool b; int c = a+b; ` |
| Add | Only allowed to add two integer variables | `int a; int c = a+ 5;` |
| Const | Cannot assign an integer const to a bool variable | `bool v1 = 2;` |
| Cond Branch | Only takes **bool** variable as inpput (and 2 labels) | `int a = 1; br a here there;` |
| Label | Label argument in control operation not present in code | `jmp itsatrap;` |
| Label | A label should be unique and not be repeated | `jmp label; label: <> .... label: <>` |
| Not | Cannot assign the output of not to an integer | `bool b = true; int a = not b;` |
| Const       | Variables cannot be redefined for a different type | `int v = 5; bool v = true;` |

These were some of the tests we used to check exhaustively for the rules developed for type checking.








