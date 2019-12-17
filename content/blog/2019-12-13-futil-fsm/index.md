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

A typical FuTIL program is shown below:

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
     (if (@ gt0 out) (enable gt0 a0 const2)
         (enable y0 const3)
         (enable z0 const4)))))
```

It is composed of structure and control parts. The structure part is straight forward: ` (new-std b0 (std_reg 32 0))` stands for instantiation of the library component `b0` with parameter `32` and `0`, and `(-> (@ a0 out) (@ gt0 left))` stands for wiring the `out` port of component `a0` with `left` port of component `gt0`. The control part specifies which components are active with `enable` keyword, and the execution logic with `par`, `seq`,`if` and`while` keywords.

In this project, we are interested in changing all the control logic to finite state machines (FSMs) and then generate simulatable Verilog program based on both FSMs and the structures.



## Design Overview

### Finite State Machine Components

In our design, an intermediate FSM is a component with `fsm_[control]` as name prefix. An FSM component has:

- input and output ports,
- connection of wires between its own ports and other components's port,
- internal control logic that determine the output signals.

The internal control logic of a FSM component can be divided into several states that determines the output signals. A state transfers to another according to some input signals. In general, all FSM components are composed of one **Start** state, some **Intermediate** states and **End** state. 

Consider the syntax `(enable A B)` . The **Start** state transfers to the **Intermediate** when the *valid* signal is high. At the **Intermediate** state, the FSM sends out valid signals to subcomponents `A` and `B`, and waits for *ready* signals from them to be high. Once both of the *ready* signals are high, the FSM transfers to **End** state and outputs *ready* signals to notify upper components. It transfers back to **Start** state when *valid* signal is low, indicating the upper components have received the *ready* signal and finished execution so it is safe for the FSM to go back to the **Start** state. The same design logic applies to all FSMs. The only difference happens in intermediate state(s): `seq` FSM has one or more intermediate states and one intermediate only transfers to next state when receiving high *ready* signal from the previous state; `if` FSM send *valid* to the module that execute the comparison and receive both *ready* and *condition* signals and determine which state it should transfer to with the *condition* signal; `while` FSM transfers to loop **Body** state when *condition* signal is high and goes to **End** State when condition is low.

<img src="fsm.pdf" style="width: 100%">

### *Read* Signals

`enable` keyword is used to determine whether a component is active. It is the easiest way of translating a program into  hardware. However, this implicitly assumes that the signal on a wire is not readable until we `enable` a component. We therefore require any data wire extra one bit to specify whether the signal is readable.

### Lookup Tables (LUT)

A component can be used more than once. For instance, if we write to register `x` more than once, we actually reused this register component. We therefore need to create a lookup table (LUT) to learn the correct input to this component.



## Implementation

### Adding FSM Passes 

We first add a pass that translates control syntax to FSM components.

Based on the design logic of FSMs, we can specify the inputs and outputs of each FSM component and the wires connecting each ports to its subcomponents. Notice we also need add *cond_read* signals to specify whether the *condition* signal from the comparison component is readable.

| FSM                  | Input Ports                                   | Output Ports                 |
| -------------------- | --------------------------------------------- | ---------------------------- |
| `enable`/`par`/`seq` | *val, rdy_A, rdy_B, ...*                      | *rdy, val_A, val_B, ...*     |
| `if`                 | *val, rdy_con, cond, cond_read, rdy_T, rdy_F* | *rdy, val_con, val_T, val_F* |
| `while`              | *val, rdy_con, cond, cond_read, rdy_body*     | *rdy, val_con, val_body*     |



### Adding LUT Passes








## Hardest Parts

1. FuTIL is implemented with [Rust](<https://www.rust-lang.org/>), so we spent some time to get familiar with the language. 
2. The design of FSM representation changes multiple times. Because the state should be store as pointer and then modified when we add transition and outputs to it. 
3. 

## Evaluation 


