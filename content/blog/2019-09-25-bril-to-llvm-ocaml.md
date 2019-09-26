+++
title = "Bril to LLVM using OCaml"
extra.author = "Daniel Weber"
extra.bio = """
  [Daniel Weber](https://github.com/Dan12) is an MEng student interested in programming languages, distributed systems, and the inner workings of computers.
"""
+++

## The Goal

For this project, I wanted to create a transformation from Bril to LLVM IR and implement this transformation in OCaml. The motivation for the first goal of the project (LLVM code generation) was to allow Bril to be compiled and run natively, instead of just interpreted. LLVM can be compiled to machine code using a tool like `clang`. Furthermore, LLVM IR supports many optimizations which allow even a naive transformation of Bril to LLVM to be quite performant. The motivation for the second goal was that OCaml provides several constructs, like variants, GADTs, and partial function application. These features might make it nicer to write IR transformations than it would to do the transformations in Typescript.

## The Implementation

### An unsuccessful representation

The first part of this project was creating some representation of the Bril IR in OCaml. One of my goals when defining a representation for Bril in OCaml was that I wanted to maintain some of the inheritance structure of the Bril definition in Typescript. For example, the Typescript definition of Bril distinguishes between effect operations and value operations with two different interfaces:

```typescript
export interface EffectOperation {
  op: "br" | "jmp" | "print" | "ret";
  args: Ident[];
}

export interface ValueOperation {
  op: "add" | "mul" | "sub" | "div" |
      "id" | "nop" |
      "eq" | "lt" | "gt" | "ge" | "le" | "not" | "and" | "or";
  args: Ident[];
  dest: Ident;
  type: Type;
}
```

This creates a nice separation between the two kinds of operations and allows a programmer to handle a generic effect operation or value operation without having to worry about which specific operation they are handling. Initially, I created a basic type that was a single variant type with a different constructor for each operation:

```ocaml
type operation =
  | Br of br_data
  | Jmp of jmp_data
  | Print of print_data
  | Ret
  | Id of un_op_data
  | Const of const_data
  ...
```

This was a simple representation and generally worked fine for something like basic block generation, when I only really cared about specifically identifying `Ret`, `Br`, and `Jmp` operations. 

One of the first things I wanted to do when I was generating LLVM code was to create a stack (since I wasn't going to implement an SSA transformation). On this stack I would map variables to stack indices, so I needed a list of all variables that were written to in a function. Using the representation I just described above, in order to extract all of the destinations from instructions I would have to write something like:

```ocaml
match op with 
| Id {dest;_}
| Const {dest; _}
| Add {dest;_}
... -> Some dest
| _ -> None
```

In typescript I could do something like:

```typescript
if (op.dest) {
    return op.dest;
} else {
    return null;
}
```

The above code does not need to know which operation it is operating on, only if the operation contains a specific piece of data.

One of the goals that I came up with for the representation was that I should have a representation that allowed me to match on an operation based on the data it contained or match on a specific operation. Furthermore, if I matched on the data in an operation, this should  statically limit the kinds of operations that I could match on. For example, if the case of a match statement I am in tells me that I have a `dest` field, OCaml should complain if I try to match the opcode of that operation with `Br`, since `Br` does not have a `dest` field.

The way I did this was by combining GADT's with polymorphic variants to create what I called constrained extensible records. The top level record type for an operation was:

```ocaml
type 'a operation = {op: 'a opcode; ex: 'a op_ex}
```

Every operation has an opcode field. Furthermore, that opcode encodes information about the type of data held in the `ex` field of the `operation` record. An opcode is a GADT that can only be constructed using a Constructor representing one of the Bril opcodes:

```ocaml
type _ opcode =
  | Jmp : [`Jmp] opcode
  | Br : [`Br] opcode
  | Add : [`Add] opcode
  ...
```

This is where the polymorphic variants come in. The `'a` in the `operation` record is a polymorphic variant representing the operation. The polymorphic variant constrains the type we can put into the `ex` field of the `operation` record. The `op_ex` type looks like:

```ocaml
type _ op_ex =
  | Effect_op : 'a effect_op -> 'a op_ex
  | Mutation_op : 'a mutation_op -> 'a op_ex
  | Nop_op : [`Nop] op_ex
  ...
```

The basic idea was that the GADT parameterized over polymorphic variants representing the opcodes would constrain the type of data that could be in the `ex` field of an operation. This kind of pattern continues down the type hierarchy and took a while to get correct. Even though this is the representation I ended up doing this project in, the reason I called it an unsuccessful representation is that in hindsight a similar effect could have been achieved by simply defining a hierarchy of plain old variants with different data structures. I thought I would be saving myself unnecessary match statements by incorporating GADTs to constrain the data but ultimately I don't think I saved myself any match cases and had to spend a lot of time wrestling with the OCaml type system.

I think there is a cautionary tale here about trying to overengineer an AST representation based on how you think you are going to use it. If anyone is interested in looking at the full type representation that I used, it can be found [here](https://github.com/Dan12/bril/blob/master/bril-ocaml/bril/bril_v2.ml). I will say though, it was a good exercise in learning about the more peculiar parts of OCaml's type system and I think made me more comfortable with GADTs and made me more cognizant about their limitations.

### LLVM Code Generation

LLVM code generation consisted of 2 main parts. First, since LLVM IR requires each basic block to have a terminator, I decided first generate all of the basic blocks of a Bril program. I could have probably simply looked for all labels with no preceding terminator in a Bril program and added a jump, but since I already written the code for creating and processing basic blocks, I decided to generate LLVM code at a basic block level.

Next, since I didn't implement an SSA transformation for Bril and because Bril variables can be overwritten, I need to create some "stack" space for all of the variables. The way I did this was by using the `alloca` LLVM instruction at the top of the function. I first collected all of the variables that were written to in a function and mapped them to an index in the stack. I also decided to make all variables the LLVM `i64` type. OCaml ints are 63 bits and I wanted to support the largest range of numbers possible. So at the beginning of a function, I inserted a call to `alloca i64, i64 n`,where `n` was the number of unique variables written to. Note, that if a variable is used without being defined anywhere in the function, this will generate a blank instruction and likely cause LLVM to fail when it typechecks the generated code.

Whenever a variable is used as part of an operation it is loaded from the stack. Whenever a variable is modified by an operation, the result of the operation is written back to the variable's location on the stack. So, for example, an add of two variables first performs a load of both variables from the stack into fresh variable names, adds the two variables together with an LLVM `add` instruction, and stores the result back to the `dest` variable of the Bril add instruction.

One other interesting aspect of this project was implementing the print function. I wrote some C code in a `helpers.c` file that just defined a `printi` and a `printb` function that printed out a 64-bit integer and a boolean `true` or `false` respectively. This C code was then compiled and linked in with the generated `.ll` file to create the final binary. In order to figure out which function to generate for each print operation, I added some code to get the type of a variable when it was defined. For example, if I saw:

```
v: int = add v w;
```

in a Bril program, I would say that variable `v` has type `int`. This may not work in general because the type of a Bril variable can technically be dynamic. For example, this is legal Bril:

```
v: int = const 1;
v: bool = const true;
print v;
```

This would actually be one potential reason to implement an SSA transformation of Bril before transforming to LLVM (some LLVM optimizations can usually optimize away the stack so for the sake of code efficiency it wasn't strictly necessary). If Bril was in SSA form, then every variable would only have 1 type and we could unambiguously figure out which print to use.

## Evaluation

### Code correctness

In order to evaluate wether I had succeeded at correctly creating code that transforms Bril into LLVM IR I wrote a battery of tests to test every possible operation and every possible edge case of every operation. The way I verified correctness was by generating the LLVM code for a Bril file, compiling and linking the LLVM code with the `helpers.c` file into a binary, running the binary and capturing the standard output of the execution. Then, I ran the same Bril code through `brili` and captured the standard output of that execution. I then compared the two standard outs to see if they agreed. 

One thing that I worried about was that I never defined the expected output of a program. So in the unlikely event that the Bril interpreter and my code have a similar bug (like a copy paste error when handling `add` and `sub`) I would not notice this issue.

I had some test programs that were written in Bril and some that were written in Typescript and compiled to Bril. The former allowed me to test some weird kinds of combinations of instructions not possible to generate by compiling Typescript. The latter allowed me to more easily write large programs that did more complicated things to stress test the code generator.

Some other evaluation metrics were considered, such as comparing the speed of a Bril program when run through the interpreter to the speed of that Bril program when compiled and run natively. However, it was decided that these results would not be very meaningful other that to confirm that running code natively is *much* faster than running code in an interpreter.

Additionally, measuring the speed at which Bril programs can be transformed to LLVM as a function of their code size would be an interesting extension to this project. We would likely prefer that this transformation take a linear amount of time in relation to the size of the input program.

### Code design

As mentioned above, one aspect where I felt like this project didn't succeed was in creating a good, strongly typed, OCaml representation of the Bril AST. Additionally, the way the LLVM code was generated was by simply generating strings of LLVM instructions for each Bril opcode. One potential way to improve upon this is to make an OCaml module that describes the LLVM AST and then write code to transform the Bril AST into the LLVM AST. Then we could just convert the LLVM AST to a string and output it to a file. 

Another potential way and one I started to explore is similar to how you can generate LLVM code in C++. You create function objects and basic block objects within those functions and then append instructions to those basic block objects. I tried to do something similar and add some static type checking when composing certain types of operations. For example, a `br` instruction takes in an `i1` argument (a boolean). So I tried to write some operation builders with some input and output type constraints. This also lead to a lot of struggling with the OCaml type system but I was able to get something reasonable working for a subset of Bril operations. You can check it out [here](https://github.com/Dan12/bril/blob/master/bril-ocaml/llvm_gen/llvm.ml)

The hardest part of this project was definitely trying to get a good representation of the Bril AST in OCaml. I think I might want to revisit this representation in future projects and try and get something that I am happy with. One interesting comparison would have been to also try to do this project in Typescript, since it has some cool type constructs.

All of the code can be found [here](https://github.com/Dan12/bril/tree/master/bril-ocaml)