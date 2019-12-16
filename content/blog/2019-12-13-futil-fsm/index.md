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

A typical FuTIL program is listed below:

```fut
(define/namespace prog
  (define/component main () ()
    ((new-std a0 (std_reg 32 0))
     (-> (@ const0 out) (@ a0 in))
     (new-std const0 (std_const 32 2))
     (new-std b0 (std_reg 32 0))
     (-> (@ const1 out) (@ b0 in))
     (new-std const1 (std_const 32 1))
     (new-std gt0 (std_gt 32))
     (-> (@ a0 out) (@ gt0 left))
     (-> (@ const2 out) (@ gt0 right))
     (new-std const2 (std_const 32 1))
     (new-std y0 (std_reg 32 0))
     (-> (@ const3 out) (@ y0 in))
     (new-std const3 (std_const 32 2))
     (new-std z0 (std_reg 32 0))
     (-> (@ const4 out) (@ z0 in))
     (new-std const4 (std_const 32 4)))
    (seq
     (par
      (enable a0 const0)
      (enable b0 const1))
     (enable gt0 a0 const2)
     (if (@ gt0 out)
         (enable y0 const3)
         (enable z0 const4)))))
```

It is composed of structure and control parts. The structure part is straight forward: ` (new-std b0 (std_reg 32 0))` stands for instantiation of the library component `b0` with parameter `32` and `0`, and `(-> (@ a0 out) (@ gt0 left))` stands for wiring the `out` port of component `a0` with `left` port of component `gt0`. The control part specifies which components are active with `enable` keyword, and the following what execution logic with `par`, `seq`,`if` and`while` logic.

In this project, we are interested in changing all the control logic to finite state machines (FSMs) and then generate simulatable Verilog program based on both FSMs and the structures.



## Design Overview





## Implementation






## Hardest Parts

1. The design of FSM representation changes multiple times. Because the state should be store as pointer and then modified when we add transition and outputs to it. 
2. 

## Evaluation 


