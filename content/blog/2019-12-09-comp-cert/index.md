+++
title = "CompCert: Formally Verified C Compiler"
extra.author = "Samuel Thomas & Daniel Weber"
extra.latex = true
extra.bio = """
  [Samuel Thomas](https://sgtpeacock.com) is a senior undergraduate applying to Graduate Schools for programming languages.
  [Daniel Weber](https://github.com/Dan12) is an MEng student interested in programming languages and distributed systems.
"""
+++

### Motivation / Introduction

### Semantic Preservation
In order for a compiler to be correct it needs to preserve the semantics of our source program. In this section, we formalize the notion of semantic correctness.

We assume that we have a formal semantics for our source and target languages that assigns *observable behaviors* to each program. We write $S \Downarrow B$ to mean $S$ executes with observable behavior $B$. An observable behavior includes things like whether the program terminates or not, various *going wrong* behaviors such as accessing an array out of bounds or invoking an undefined operation like dividing by zero. It also includes a trace of all external calls (system calls). However, it doesn't include the state of memory.

A possible definition of semantic preservation is, with $S$ being a source program and $C$ being a compiled program:

$\forall B, S \Downarrow B \Leftrightarrow C \Downarrow B$

This definition is too strict because a compiler might want to optimize away certain *going wrong* behaviors if, for example, they come from dead code. For this reason, we want to give the compiler a little bit more freedom and so the following definition is preferable:

$S \texttt{safe} \Rightarrow (\forall B, C \Downarrow B \Rightarrow S \Downarrow B)$

$S \texttt{safe}$ means that $S$ doesn't go wrong. This definition is saying that all of the observable behaviors of $C$ are a subset of the observable behaviors of $S$ and that if $S$ doesn't go wrong, then $C$ doesn't go wrong.

#### Verification vs. Validation
The paper models a compiler as a total function from source programs to either `OK(C)`, a compiled program, or `Error`, the output that represents a compile-time error, signifying that the compiler was unable to produce code. There are two approaches for establishing that a compiler has the semantic preservation property discussed above: verifying the compiler directly using formal methods or verifying a *validator*, a boolean function accompanying the compiler that verifies the output of the compiler separately. The second approach is convenient because sometimes the validator is significantly simpler than the compiler. We'll see this approach used later for verifying part of the register pass.

### Structure of the Compiler

The source language of the CompCert compiler is Clight, which is a subset of C that includes most familiar C programming constructs like pointers, arrays, structs, if/then statements and loops. The compiler front end consists of an unverified parser that parses a source file to a Clight AST. From there, the formally verified section of the compiler performs several passes that repeatedly simplify and transform the representation of the source code all the way to PowerPC assembly code. Then, an unverified assembler and linker take the assembly code and generate an executable that can be run.

In total, CompCert formally defines 8 intermediate languages and 14 passes over them. The 14 passes must be proven to preserve the semantics of the original program. The first few passes simplify the C code by converting all types to either ints or floats (pointers get converted to ints) and explicitly describing memory accesses. The result of these passes is an intermediate representation called Cminor, and below is an example of a translation from Clight to Cminor.

<img src="transf_ex.png" style="width: 100%">

As you can see, function signatures have been made explicit, implicit casts have been made explicit (like the cast from float to int), array accesses have been transformed into exact byte offsets from pointers, and the size of the function's activation record has been made explicit (this is for dealing with the address of operator, which requires function local variables to be mapped to a location on the stack frame of the function).

Next, CompCert performs instruction selection for the specific architecture that is targeting. This is done via instruction tiling which can recognize basic algebraic identities. Next, comp CERT translates C minor sel into RTL which represents control flow in a CFG. This is a convenient representation to perform optimizations on. In this representation, comp CERT is able to run several dataflow analyses on the program in order to perform optimizations such as constant propagation, common sub expression elimination, and lazy code motion. Lazy code motion was limited using the validator approach, likely due to its complexity, while the other two where written in coq and formally proven correct. The RTL representation still uses pseudo registers to store values. The next transformation past executed by comp CERT is to allocate the pseudo registers two hardware registers using a register allocation pass. The register allocation pass also uses the validator method, with the coloring algorithm implemented oh camel. Further passes linear rise the CFG, spill registers on the stack and insert the necessary loads from temporaries, and creating function prologue and epilogues. Finally, comp CERT performs Instruction scheduling to increase instruction level parallelism on super scaler processors, such as the powerpc architecture, and generates PowerPC assembly code. All of these transformations are formally verified to preserve the semantics of the original program.

### Verification of the Register Allocation Pass

### Evaluation
