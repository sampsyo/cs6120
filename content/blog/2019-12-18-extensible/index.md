+++
title = "Composable Brili Extensions"
extra.author = "Daniel Weber"
extra.bio = """
  [Daniel Weber](https://github.com/Dan12) is an MEng student interested in programming languages and distributed systems.
"""
+++

## The Goal

The first project for this course was to extend the [Bril language](https://github.com/sampsyo/bril) in any way that we wanted. The initial Bril code base included a parser from a textual representation of Bril to an AST of the source program encoded in JSON. It also included a program that transforms Typescript programs into Bril programs and an interpreter for the language once it was in JSON format. Some projects added new standalone extensions to the code base such as a [Bril debugger](https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/brildb/) or a [Bril to C translator](https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/bril-c-backend/). As long as these projects don't make any changes to the underlying grammar or representation and only add to the existing codebase instead of modifying it, they can be relatively easily merged into the code base.

However, some projects created extensions to the Bril language that added new operations, new state, and new control flow to the language. These projects include adding [record types](https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/recordtypes/), [dynamically allocated memory](https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/manually-managed-memory/), and [function calls](https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/function-calls/) to the language. All of these projects make changes to the grammar of the language by adding new types of operations. Additionally, all of the project modify the interpreter to support their new operations. All of these changes require adding additional state to the evaluation function, which all conflict with each other.

In this project, I designed and implemented a composable language extension system for the Bril interpreter that allows developers to independently create language extensions that can later be composed together. For example, consider an extension A that adds a new operation 'a' to the language and an extension B that adds a new operation 'b' to the language. Once these two extensions are merged into the codebase, other developers should be able to compose extensions A and B together to create an interpreter that supports both operations 'a' and 'b'. Furthermore, the developers of extension A and extension B should ideally never have to know about each other's existence.

First, I will discuss the implementation in its full detail. Then, I will discuss why I made some of the design decisions that I did. Finally, I will highlight some of the issues with the design and some lessons learned.

## The Implementation

I implemented the extensible interpreter in Typescript. I first identified some common datatypes that all extensions would use. The first was the base instruction interface. I decided that all instructions would be identifiable by operation field called op (sometimes called a discriminant):

```javascript
interface BaseInstruction {
  op: string;
}
```

Next, I decided that all functions in Bril would have a name and a list of instructions or labels:

```javascript
interface BaseFunction<I extends BaseInstruction> {
  name: Ident;
  instrs: (I | Label)[];
}
```

One interesting thing to note is that the `BaseFunction` interface is generic over the type of instructions that it will be containing. Finally, a Bril program contained a list of functions:

```javascript
interface BaseProgram<I extends BaseInstruction, F extends BaseFunction<I>> {
  functions: F[];
}
```

These three interfaces define the most basic structure that a Bril program can have. All extensions must be extensions to Bril that respect this generic language definition. Put another way, every Bril extension must work on a subtype of the `BaseProgram` interface parameterized over some `I` and `F` that extend `BaseInstruction` and `BaseFunction` respectively.

Next, I defined a generic evaluation function for evaluating instructions:

```javascript
type evalInstr<A, PS, FS, I> = (instr: I, programState: PS, functionState: FS) => A;
```

The `evalInstr` function is parameterized over 4 types. The first type, `A`, is the action type. Every evaluation function needs to generate an action that specifies which instruction to execute next. The second and third types, `PS` and `FS`, represent the program state and the function state respectively. The program state type is meant to represent the entire state of the currently running Bril program (so think things like global variables). The function state holds only function local state for the currently executing function (like the values of local variables). The final parameter is `I`, which is the type of the instruction the evaluation function operates over.

Each extension defines an `evalInstr` function that specifies how to update the program and function state in terms of the instructions it adds/extends as well as which action to generate for those instructions. In order to compose these functions, each extension defines a function that takes in a function of type `evalInstr` and returns a function of type `evalInstr`. The idea is that the function that is returned has 2 cases. In the first case the instruction passed in is of the type that this extension is extending/adding. In this case the function executes the logic to update the program and function states and return an action according to the operation. In the other case, the instruction is not an instruction that this extension implements. So it dispatches to the `evalInstr` function that was passed in to the original function. The code looks something like this:

```javascript
function evalInstr<A,PS extends ProgramState,FS extends FunctionState, I extends BaseInstruction>(
    baseEval: (instr: I, programState:PS, functionState:FS) => A
) {
    return (instr: bril.Instruction | I, programState:PS, functionState:FS): A | brili_base.Action => {
        if (isExtInstr(instr)) {
            return handleExtInstr(instr);
        } else {
            return baseEval(instr, programState, functionState);
        }
    }
}
```

Each extension also usually defines types `ProgramState` and `FunctionState`, which are record types representing the types of fields that this extension expects to be in the program state and the function state respectively. For example, if

```javascript
type FunctionState = {env: Env};
```

then that means that this extension expects the function state to have an `env` field of type `Env` (this is the mapping from variable names to values in the base version of Bril). Because `FunctionState` is a record type, the restriction `FS extends FunctionState` implies that any object of type `FS` must have at least the same fields with the same types as specified in `FunctionState`. In the above case, `FS` must be a record that has the `env` field. It may have more, but it has at least that one field. This is simply how record subtyping works in Typescript.

Composition of extensions now is a simple matter to composing the `evalInstr` functions together. I implemented a `Composer` class that is used to compose extensions together which composes extensions in the following way:

```javascript
constructor(evalExts: ((baseEval: evalFunc<A, PS, FS, I>) => evalFunc<A, PS, FS, I>)[], ...) {
    this.evalInstr = (instr,programState,functionState) => {
        throw `unhandled instruction: ${instr.op}`;
    }
    for (let ext of evalExts) {
        this.evalInstr = ext(this.evalInstr);
    }

    ...
}
```

It simply creates a base `evalInstr` function that throws an unhandled exception and then in a loop composes all of the `evalInstr` functions in `evalExts`.

In addition to defining how different operations are handled, an extension may also want to override/extend the handling of actions generated by evaluating the various operations. First, however, I need to define how control flow is handled in Bril programs. When a Bril program is executing there is a program counter variable called `pc` that keeps track of where we are in the program. The type of this `pc` variable is:

```javascript
type PC<I extends BaseInstruction, F extends BaseFunction<I>> = { function: F; index: number };
```

It contains a `function` field which specifies which Bril function we are currently in and an `index`, which specifies which instruction in that function we are executing. These are again parameterized over an instruction and function type in order to support many different kinds of extensions to functions.

The handlers for actions take in the action generated by the current instruction and the current PC and generate a new PC. We also supply the action handler functions the current function and program state in case action handler extensions want access to the current program state. The type of these action handler functions is thus:

```javascript
type actionHandler<A, PS, FS, I extends BaseInstruction, F extends BaseFunction<I>> = (action: A, pc: PC<I, F>, programState: PS, functionState: FS) => PC<I, F>;
```

These handlers are also composable in a very similar manner to the `evalInstr` functions. The extensions export a function that takes in an `actionHandler` function that represents the action handler function being extended and outputs an `actionHandler` function. The outputted function executes the extension's action handling functions if the current action is one that it handles and otherwise dispatches to the action handling function that it was extending. In the `Composer` class I do the following, which is very similar to how I composed `evalInstr` functions:

```javascript
constructor(..., actionHandleExts: ((extFunc: actionHandler<A, P, FS, I, F>) => actionHandler<A, P, FS, I, F>)[], ...) {
    this.handleAction = (action,pc,programState,functionState) => {
        throw `unhandled action`;
    };
    for (let ext of actionHandleExts) {
        this.handleAction = ext(this.handleAction);
    }
    ...
 }
```

Finally, the composer exposes a program that evaluates the 
