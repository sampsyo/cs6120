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

Bril currently supports 2 types, `int` and `bool`, which makes type checking relatively easy. Our type checker is defined such that an arithmetic operation like this:

    a: int = add b c

would raise an error if either `a` or `b` has type other than `int`. Similarly, boolean operations only accept all bool arguments and comparison operations have integer arguments but a boolean destination. These type definitions have been nicely defined [here](https://capra.cs.cornell.edu/bril/langref.html). During type checking, we also ensure that the variables in the instruction have been defined before in the program order. Conversely, we make sure that there isn't a redefinition of the variable using different types. For control flow operations, we ensure that labels are present in the code and are uniquely defined. This allows us to treat label strings as a separate type, invisible to the user. 

The error raised by the type checker outputs the line at which we find the first operation breaking the type check rule and mentions the type of error found.


### Implementation

We have a first pass in the algorithm which collects a list of labels and ensures that each label is unique. If there are multiple labels with the same name, we throw an error at this point. This also helps us create a set of label names which is used to check for valid strings (of labels) during the second pass for control flow operations.  In the second pass, we go over each instruction and check for various type errors. Some fundamental checks in this process are:

1. Invalid instructions: This ensures that all the arguments and destination (if applicable) of an instruction are available, which is given by the number of arguments.

2. Argument and destination type: For various operations we check if arguments and destination variables have the correct type. Something like `d: bool = add a b` where `a` and `b` are integers would raise an error for the destination variable.

3. Redefined variables: We check if the destination variable has already been assigned to a different type in that context.  Hence the following set of instructions:

   `a: int = const 2; `

   `a: bool = const true;` 

   are not allowed but redefinitions on the same type is definitely possible:

   `a: int = const 2;` 

   `a: int = const 5;`

   We do this by keeping a set of variables of each type (`int` and `bool`) defined in the function. This helps while checking existing definitions and possible redefinition errors.

4. Undefined variables: We check if the arguments to the instruction have defined variables using the set of variables mentioned before. A simple example would be an instruction like `a: int = const 5; c: int = add a b` where `b` was not defined before the instruction.

   Note: We haven't taken jumps / branches into consideration which implies if there was a jump instruction before where the variable was defined, we would still throw an error. In hindsight, this can be resolved by traversing the basic blocks in a topological order.



## Hardest parts during the implementation

The type checker implementation, though straightforward, had a few challenges.
1. The typing rules need to be designed carefully for each operation, including arithmetic, boolean, and control instructions. Take the following branch instruction as an example: `br b left right`. The type checker needs to go through all the existing boolean variables to ensure that `b` is predefined and `left` and `right` are valid labels in the code snippet. Also, though currently only two basic types are supported in Bril, there could be extensions for more types like list, stack etc. Thus, it is important to maintain modularity of arithmetic/boolean and control flow checking so that future updates could be made easily to support them.
2. To ensure that the type checker works properly with labels, a first pass of the labels is designed. Through the first pass, all existing labels are saved and no duplicate ones are allowed to avoid conflicts. Though this almost doubles the type checking overhead, it further ensures the correctness of the program by static analysis.
3. Keeping track of the line number and returning proper error message when encountered with an type checking error. To track the line number, the type checker works on the Bril text representation and is developed with its own parser. When scanning through the code line by line after the first pass, a line number is maintained during type checking to help the programmer debug. In order to maintain the line number with blank lines and comments, we take in the original code snippet and ignore these lines while updating it during type checking.



## Evaluation and results

We wrote a set of benchmarking programs to test our implementation for various types of rules defined in the design section above.
The general test cases are classified into two sub-directories: [should-fail](https://github.com/tissue3/bril/tree/master/test/type-check/should-fail) and [should-pass](https://github.com/tissue3/bril/tree/master/test/type-check/should-pass). It has expected output named as `*.out` corresponding to the input file `*.bril`. The user can simply run all test cases in a directory by running `turnt directory/*.bril`. For example, to run should-fail benchmark, one can just run `turnt test/type-check/should-pass/*.bril` at the main directory of bril.
Because should-pass cases are trivial, where we enumerate list existing operations, in the following table we only list tests cases that our type checker will report an error message. The second column of the table aims to help one understand why an error would be reported by our type checker.

| Instruction | Type Checking Rule |Testing Code Snippet  |
| ----------- | :-----------: | :-------------: |
| Conflict Definition  |   Variables cannot be redefined with a different type.   |      v: int = 5;<br/> v :bool = true;      |
| Arithmetic         |          Adding an integer and a boolean type.           |                a: int = 4;<br/>b: bool = true;<br/>c: int = add a b;               |
| Boolean         |      Cannot assign the output of boolean to an integer.      |    b: bool = true;<br/> a:int = not b;    |
| Const       |    Cannot assign an integer const to a bool variable.    |            a: int =  const true;             |
| Cond Branch |    Only takes bool variable as input (and 2 labels).     |     a: int = 1;<br/> br a here there;<br/> here: print a;<br/>there: jmp here; |

Finally, though not required, we also implemented other checking passes as long as an input stream is parsable by bril2json. For example, `v0: boolean =const true;` is a legal statement for bril2json, but boolean is not an existing type in bril, so our type checker will report the error. Such kind of checking pass includes type existence checking, number checking, argument existence checking, label existence checking and label repentance checking.

| Instruction |               Type Checking Rule               |          Testing Code Snippet          |
| ----------- | :-----------------------------------------------------: | :-----------------------------------: |
| Type existence       | The destination type is undefined.          | v0: boolean = const true; |
| Argument Number       | The expected argument is 2 but only 1 is given.          |  v0: int = const 5;<br/> v1: int = add v0;  |
| Argument Existence | The argument is never defined. | v1: int = add v0; |
| Label Existence      | Label argument in control operation not present in code. |            jmp its_a_trap; (Consider this as is a full program)     |
| Repeated Label |      A label should be unique and not be repeated.       | jmp label;<br/> label: a:int=1;<br/> label: a:int =2 |

By and large, we have implemented the checker satisfying all of our defined behaviors. But we don't know if that's exhaustive for all possible errors (not necessarily type errors). We would be very happy if someone comes up with more cases and reaches out us by mail or GitHub issues.





