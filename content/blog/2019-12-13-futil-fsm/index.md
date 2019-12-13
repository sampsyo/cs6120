+++
title = "Runtime Execution Profiling using LLVM"
extra.author = "Yi Jing, Zhijing Li, Neil Adit, Kenneth Fang, Sam Thomas"
extra.bio = """
[Zhijing Li](https://tissue3.github.io/) is a 2nd year Phd student at ECE department, Cornell. She is interested in compiler optimization and domain-specific languages. 
[Yi Jiang](http://www.cs.cornell.edu/~yijiang/) is a 2nd year Phd student in CS dept, Cornell. She is interested in designing compilers for Processing-In-Memory related architectures and also general compiler optimization. \n
[Neil Adit](http://www.neiladit.com) is (guess what!) also a 2nd year PhD student at CSL, Cornell. He is interested in fun things in compiler research.

"""
extra.latex = true

+++



## Goal


The goal of this project to generate Control FSM for FuTIL, which can be divided into the following two parts:

- Convert a Control AST in FuTIL to an intermediate FSM structure
- Generate RTL from the intermediate FSM structure

## Background

FuTIL is an intermediate language that represents hardware as a combination of *structure* and *control*. The *structure* represents how subcomponents are instanced and wired together, while the *control* determines how these subcomponents are activated at different times. The ultimate goal of FuTIL is to provide an RTL backend for the Dahlia language. The structure is fairly straightforward to convert to RTL, which easily represents components and wires, but the control flow statements have no straightforward representation in RTL. We will convert the control statements in FuTIL into an RTL FSM to facilitate RTL generation from FuTIL.



## Design Overview





## Implementation






## Hardest Parts

1. The design of FSM representation changes multiple times. Because the state should be store as pointer and then modified when we add transition and outputs to it. 
2. 

## Evaluation 


