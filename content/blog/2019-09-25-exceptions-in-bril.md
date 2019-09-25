+++
title = "Exceptions in Bril"
extra.author = "Rolph Recto"
extra.bio = """
[Rolph Recto](rolph-recto.github.io) is a third-year graduate student studying
the intersection of programming languages, security, and distributed systems.
"""
+++


## Overview

We add exceptions as a control structure in Bril to support non-local
control flow. 
This allows frontends to compile exceptions directly.
The ultimate goal is to extend Bril further with more powerful control
structures such as first-class continuations or algebraic effects.


## Function Calls

To make the implementation of exceptions nontrivial and to make exceptions
actually be non-local control structures, we first extend Bril so that it
supports function calls.
We do this by adding a `call` instruction that takes as arguments a
function name and a list of function arguments.
We also extended function declarations to allow for formal parameters.
The Bril text format now allows for declarations of the form

```
funcName arg1:type1 ... argN:typeN { instrs }
```

Our implemention explicitly represents the program stack as an object. 

The interpreter has four configuration variables that consist of its current
activation record:

* Bril function object --- the current function whose list of instructions
  is being executed

* Variable environment --- map from local variables to values

* Handler environment --- map from exception handlers to (see below)

* PC index --- the index into the instruction list of the current instruction
  being executed

The interpreter proceeds by executing instructions, which either change
environment mappings and/or sets the PC index of the next instruction
to execute.
When `call` is executed, the interpreter does the following:

* The current activation record is pushed into the program stack.

* An empty variable environment is constructed, and the formal parameters
  of the function are mapped to the arguments of the `call` instruction.
  An empty handler environment is constructed.

* The PC index is changed to 0, and the variable and handler environments 
  are updated.
  The interpreter resumes execution in this new state.

When a function returns, it pops the top of the program stack as the new
activation record or, if the program stack is empty, ends program execution.


## Exceptions

Once we added support for function calls, implementing exceptions came quite
easily.
We add two instructions to this end: `handle exnName handlerLabel`
installs a handler for exception `exnName` in the handler environment,
while `throw exnName` throws an exception with name `exnName`.
When an exception `exnName` is called, the interpreter does
the following:

* The interpreter performs *stack unwinding*: first, it checks if there
  is a handler for `exnName` in the current handler environment.
  If none exists, it pops the top of the program stack and checks for a handler
  in that activation frame.
  It does this until it finds an appropriate handler; otherwise, it throws
  a exception (in the metalanguage) since a Bril exception was not handled
  properly.

* Once it finds an appropriate handler, the interpreter sets the
  handler's activation record as its current activation record and then
  it jumps to the PC index of the handler label.


## Apologia

Our implementation explicitly represents the program stack.
Our implementation could also have implicitly represented the program stack
using the interpreter's own stack by making recursive calls to 
`evalFunc`, but we obviated this design for the more explicit one to have
finer control over the program stack of Bril and to support
future development of other non-local control structures that
manipulate the stack.
(For example, first-class continuations would involve storing program stacks
as values in the environment map.)
Also, the implicit representation of program stacks would essentially fix
the non-local control structures of Bril to the control structures
the metalanguage has.
With the implicit representation, to throw exceptions in Bril programs
one would need to throw an exception in the interpreter as well;
supporting first-class continuations or algebraic effects would be impossible
in this way.

Our implementation of exception handlers does not support passing
exception objects.
To support this in the future, we make handlers be standalone functions instead
of just labels in an existing function.
This would make reasoning about the control flow of the program even less local
since it would make handlers not syntactically part of the contexts in which
they might be thrown, so we obviated this design choice for the current
implementation.

Another limitation of the current implementation of exceptions is the fact that
handlers are not tied to a lexical scope. 
That means a handler would override any handler for the same exception name
installed at the same function.
We choose this simpler albeit somewhat unintuitive design to obviate the
introduction of nested lexical scopes in Bril, which we believe is
antithetical to its intended use as a simple intermediate language.


## Evaluation

Our main goal for evaluation is to check whether our implementation is correct.
We considered performance considerations as out of bounds for evaluation since
the Bril interpreter is sufficiently different enough from how, say, ISA
instructions are implemented in hardware that we cannot infer anything useful
regarding the performance of an analogous implemenation of our exceptions
mechanism in the latter from results in the former.

To evaluate correctness, we created a suite of tests that check a variety of
situations in which exceptions might be used.
The suite has both positive and negative test cases to check for normal use of
exceptions and for when they are used improperly.
An inexhaustive list of tests include:

* (Positive) No stack unwinding: thrown exception is handled by a handler 
  installed in the current activation frame

* (Positive) Deep stack unwinding: thrown exception is handled by a handler
  high up in the stack

* (Negative) Unhandled exception: no handler is installed for an exception

Using the Turnt testing tool, we were able to verify that our implementation
passes all the tests in the suite.



