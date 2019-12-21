+++
title = "Composable Brili Extensions"
extra.author = "Daniel Weber"
extra.bio = """
  [Daniel Weber](https://github.com/Dan12) is an MEng student interested in programming languages and distributed systems.
"""
+++

## The Goal

The first project for this course was to extend the [Bril language](https://github.com/sampsyo/bril) in any way that we wanted. The initial Bril codebase included a parser from a textual representation of Bril to an AST of the source program encoded in JSON. It also included a program that transforms TypeScript programs into Bril programs and an interpreter for the language once it was in JSON format. Some projects added new standalone extensions to the codebase such as a [Bril debugger](https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/brildb/) or a [Bril to C translator](https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/bril-c-backend/). As long as these projects don't make any changes to the underlying grammar or representation and only add new files to the existing codebase, then they can be relatively easily merged into the codebase.

However, some projects created extensions to the Bril language that added new operations, new state, and new control flow to the language. These projects include adding [record types](https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/recordtypes/), [dynamically allocated memory](https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/manually-managed-memory/), and [function calls](https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/function-calls/) to the language. All of these projects make changes to the grammar of the language by adding new types of operations. Additionally, all of the project modify the interpreter to support their new operations. All of these changes require adding additional state to the evaluation function, which all conflict with each other.

In this project, I designed and implemented a composable language extension system for the Bril interpreter that allows developers to independently create language extensions that can later be composed together. For example, consider an extension A that adds a new operation 'a' to the language and an extension B that adds a new operation 'b' to the language. Once these two extensions are merged into the codebase, other developers should be able to compose extensions A and B together to create an interpreter that supports both operations 'a' and 'b'. Furthermore, the developers of extension A and extension B should ideally never have to know about each other's existence.

This is different from an extensible language framework, such as [Polyglot](https://www.cs.cornell.edu/andru/papers/polyglot.pdf), because language extensions in Polyglot explicitly specify the language that they are extending. With composable extensions, the goal is that the language extension simply defines a small set of assumptions about the abstract language that it is extending and then implements its extensions based on those assumptions. Then, any base language that satisfies those assumptions can be extended by the extension.

## The Implementation

I implemented the extensible interpreter in TypeScript. I first identified some common datatypes that all extensions would use. The first was the base instruction interface. I decided that all instructions would be identifiable by operation field called op (sometimes called a discriminant):

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

then that means that this extension expects the function state to have an `env` field of type `Env` (this is the mapping from variable names to values in the base version of Bril). Because `FunctionState` is a record type, the restriction `FS extends FunctionState` implies that any object of type `FS` must have at least the same fields with the same types as specified in `FunctionState`. In the above case, `FS` must be a record that has the `env` field. It may have more, but it has at least that one field. This is because TypeScript uses structural subtyping for records (as opposed opposed to nominal subtyping).

Composition of extensions now is a simple matter of composing the `evalInstr` functions together. I implemented a `Composer` class that is used to compose extensions together which composes extensions in the following way:

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

Finally, the composer exposes a function that evaluates a Bril program object which extends the `BaseProgram` described above:

```javascript
evalProg<Prog extends BaseProgram<I,F>>(prog: Prog)
```

This function finds the main function object and then creates a new `PC` object with that function set as its function field and the index field set to 0. Then, in a loop it gets the current instruction, executes it and gets back an action, and then updates the `pc` based on that action. It repeats this until the index of the pc goes out of the current function's bounds. The `Composer` class also takes in two functions `initP` and `initF`, which initialize the program and function states respectively.

In order to create a new Bril interpreter which is the composition of extensions A, B, and C, you simply need to define a function state that contains all the fields required by all the function states in all of the extensions, a program state that contains all of the fields required by all of the program states in the extensions, and initialization functions for those types. Then, you simply create a new instance of the `Composer` class with the extensions and action handlers you want.

## Example Extensions

### Bril Base

In order to demonstrate the usability of this system, I implemented a few extensions and then composed them together. First, I implemented the base Bril language as an extension. For the `evalInstr` function, this mostly just involved copying over the switch statement on the instruction operation from the base implementation and adding an `env` field to the function state. Then, the `actionHandler` function performed the same logic as in the `evalFunc` function in the current brili code except that variable `i` got replaced with the `pc`. I also had to add checks to each of the two new functions to make sure that the instruction/action being handled was actually one of the instructions that the function was meant to handle. Otherwise, it would just dispatch to the base instruction evaluation or action handler function that was passed in (similar to how it was described above). Implementing this base extension required very few changes to the code that was in the `brili.ts` file. Most of it was simply copied over and a few minor tweaks were made.

### Manually Managed Memory

The next extension that was implemented was the [manually managed memory](https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/manually-managed-memory/) extension. This extension added a heap datastructure to the program state and a way to allocate space on that heap. It also added a new value type, a pointer, which pointed to values in the heap. The new operations added by this extension can also load values from the heap into variable and store the value in a variable into the heap. This also requires an environment field in the function state. This leads to the following definitions in the code for this extension:

```javascript
type ProgramState = {heap: Heap<Value>};
type FunctionState = {env: Env};
```

After declaring these definitions, creating the rest of the extension was simply a matter of copying over the code that handled the new operations from that project into the `evalInstr` function for that extension. I didn't need to define an `actionHandler` function because this extension did not add any additional control flow to the base Bril language. Because of this assumption, and the assumption that there was an environment field that was being managed in the function state, this extension did kind of need to know that there was going to be an underlying base Bril language that already implemented the control flow and correctly maintained the environment.

### Record Types

The third extension that I ported over was the [record types](https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/recordtypes/) extension. This extension added record types to the language, as well as operations to create records and access/set record fields. This extension added a `typeEnv` function state variable to their extension to keep track of the defined record types in a function. Similar to the memory extension, it also assumed that there was some kind of local variable environment, so the function type for this extension looked like:

```javascript
type ProgramState = {};
type FunctionState = {env: Env, typeEnv: TypeEnv};
```

This extension also didn't require any additional control flow compared to the base Bril language so I also didn't implement an `actionHandler` function.

### Function Calls

The final extension was to add function calls. This extension was interesting because it added new control flow to the Bril language. There were two different projects that both added function calls to the language: [Function Calls and Property-Based Testing](https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/function-calls/) and [Exceptions in Bril](https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/exceptions-in-bril/). The Function Calls and Property-Based Testing project added function calls by recursively calling the `evalFunc` function and augmenting the behavior of the `ret` instruction to have `evalFunc` return a value. This approach didn't really fit that well into the new framework that I had build, because in my framework control flow revolves around the `pc` variable. The second project primarily added exceptions, but in order to add exceptions it also added basic support for function calls. This project's method of adding function calls involved creating new stack frames for each function call and pushing and popping the stack frames to implement call and return instructions. This method was more in line with the way my framework handled state, so I decided to just add the function call part of the exceptions project.

This extension required adding several additional fields to both the program and function states. The main thing that was added was an array of stack frames (i.e., a stack) in order to correctly handle function calls and returns. Each stack frame holds the function state and the return pc:

```javascript
type FunctionState = { env: Env };

type StackFrame<FS extends FunctionState, I extends BaseInstruction, F extends BaseFunction<I>> = [FS, PC<I,F>];

export type ProgramState<FS extends FunctionState, I extends BaseInstruction, F extends Function<I>> = { functions: F[], currentFunctionState: FS, callStack: StackFrame<FS,I,F>[], initF: () => FS };
```

Furthermore, arguments were added to the function records. These arguments consist of a name and a type:

```javascript
interface Function<I extends BaseInstruction> {
    name: bril.Ident;
    args: Argument[];
    instrs: (I | Label)[];
}
```

The only additional operation added in this extension was the `call` operation. The `evalInstr` function for this extension was quite simple: it just returned a new call action with the name of the function being called and the argument variables being passed into the function. Most of the control flow logic was handled in the `actionHandler` function exported by this extension. On a call action, this handler created a new stack frame with the `initF` function provided in the program state record and then pushed the current stack frame onto the call stack. It also made the assumption that all of the `evalInstr` functions were called with the program state `currentFunctionState` field. This was indeed the case, as the main loop of the `Composer` class contained this code:

```javascript
let action = this.evalInstr(line, programState, programState.currentFunctionState);
pc = this.handleAction(action, pc, programState, programState.currentFunctionState);
```

This allowed the `actionHandler` function to set the new function state on a function call. In addition, this extension overrides the `end` action. This action is usually generated by the `ret` operation in the base Bril implementation. However, instead of terminating the program, this extension modified the behavior to pop the current stack frame off of the program state's `callStack` field, set the `currentFunctionState` field to that popped stack frame, and finally setting the pc to the return pc in the popped stack frame.

This extension required a bit more effort than the other extensions because of the changes in control flow made to the interpreter. However, much of the logic could simply be copied over from the exceptions project with only minor tweaks.

### Brining it all together

In order to test the extensions and their composition I created a new instance of the `Composer` class with the 4 `evalInstr` functions from each of the extensions as well as the 2 `actionHandler` functions from the base and the function call extensions. I then added all of the test cases from each of the extensions to the project and ran the composed interpreter over all of the test cases, which all passed (there was one minor bug but it was easily fixed). I also added a test case that combined some operations from each extension, which was also correct.

## Evaluation of the system

### Extension development

One of the goals that I had when developing this system was that extensions should be able to be developed in isolation. In evaluating this goal, I will look back at the four extensions that I implemented. For the most part, each extension could be developed in isolation as long as each extension specified the assumptions it was making about the state. For example, the base extension specified that the function state had to include an environment field that stored all of the function local variables. However, because of the use of generics, the base extension never specifically names a type that the function state must be. Instead, it leaves it up to the composer of the extensions. Furthermore, even though the base extension assumes that function state has an environment field, it is kind of implicitly assuming that this field will be used correctly by all other extensions that are composed with this extension. For example, if an extension comes along and lowercases all variable names before getting and setting to the environment, it could potentially interfere with the base extension. Or, if there is a different extension that assumes that function local data is stored in a field called `local_vars` instead of `env`, then these two extensions may not be properly composable.

This was a more general trend that I realized while developing the extensions. Extensions be developed in relative isolation using the framework. However, when the extensions are composed together, the composer needs to be aware of all assumptions, implicit and explict, that are made by each extension in order to determine if the extensions can be composed in a meaningful way. On example is that there is some code in the manually managed memory code that assumes that a value is either a number, a boolean, or a pointer. However, when we compose this extension with the record type extension we have values that can be record types as well. Similarly, there is code in the record types extension that assumes that a value is either a number, a boolean, or a record. While these two extensions can operate in isolation in the same program, if I try and create a record with a pointer field things start to break.

In order to have extensions be correctly composable with each other there need to be some conventions that extension developers need to follow. To solve the above example, if the two extensions were a bit more generic with the types of values they assumed in the environment, then they could probably be correctly composable. In other words, while extensions can be developed in isolation, they need to be aware that they will be extended.

The ease at which extensions could be developed was also an important feature that I wanted to have. The minimal number of changes that I had to make to the code for the base, memory, and record type extensions indicate that writing these extensions is not that much more challenging than writing the extensions by simply adding code to the original Bril interpreter. 

The main challenge then remains in actually composing the extensions together. However, all of the Bril extensions assumed only the base extension, so multiple versions of the interpreter can easily be made by simply composing the base extension with the extension developed for the project. Then, all of the interpreter code can be merged into the repository in a conflict-free way, because of the isolation of the extensions from each other. There is still another challenge that needs to be solved, which is the extensibility of the parser. This may be solved by rewriting the parser using [parser combinators](https://en.wikipedia.org/wiki/Parser_combinator). However, I have not explored this option in any great depth.

### Type safety

Another goal that I had for this project was that when composing extensions, the composer should be assisted as much as possible by the type checker. In TypeScript, you can basically revert to untyped Javascript by simply making everything have type `any`. If all of the generic types were removed and replaced with the all-encompassing `any`, then it would be very easy to compose two extensions because their function signatures trivially match. However, you lose some type safety, such as the restrictions placed on the function and program state by each extension. So I added in generics for the program and function states. This is able to catch some type errors at compile time. However, when there is a type error the generics usually generate quite verbose type errors from the compiler, which can sometimes be difficult to parse through, especially because of all of the restrictions placed on the generics (such as the `extends` conditions). 

The generics for the instruction and function types were also added to increase type safety within the main evaluation loop and within the handler functions. Having the `pc` parameterized on those types allowed me to write code that knew that each function would have a list of instruction and a name. However, because the Bril program is parsed from a JSON file and then cast to a Bril program type without any dynamic checks means that this type safety was largely negated by the initial cast. More fundamentally, the way most extensions check that the instruction passed into `evalInstrs` is pretty unsound. The `instr` type that `evalInstrs` usually takes in is given the following type:

```javascript
instr: Instruction | I
```

where `I` is the generic type and `Instruction` is the type of the instructions that the extension implements. The `isInstruction` function is usually just a function that returns a type guard if the opcode of the instruction matches one of the implemented instruction ops (here in the `instrOp` array):

```javascript
function isInstruction(instr: BaseInstruction): instr is Instruction {
  return instrOps.some(op => op === instr.op);
}
```

This is not really a safe type guard because one of your instructions could assume a `dest` field in the instruction but this is only checking that the opcode matches, not that the instruction has all of the correct fields. This could maybe be improved by more comprehensive run time type checking, but would still probably not rule out everything that could go wrong. Furthermore, it could do the wrong thing if extensions are composed in the wrong way. If two extensions `A` and `B` implement operation `a` but an extension `A` removes some fields of operation `a`, if `B` extends `A`, then `B` will be operating on the wrong kind of instruction. However, if `A` extends `B`, then the `a` operation will be handled by extension `A` and never be propagated to `B`.

Furthermore, one other issue with the above code is that the `instrOps` array might not contain all of the correct operations. This is actually a bug that did run into because I forgot to include the `"const"` field in the base extensions `instrOps` array. If the type `Instruction` is the discriminated union of all of the instructions supported by an extension then `Instruction["op"]` is the discriminated union of all of the opcodes of all of the instructions. Ideally, we would want to transform the type `Instruction["op"]` into to `instrOps` array statically. However, TypeScript doesn't really support this transformation. However, you can convert const arrays to discriminated unions, so I came up with the following code snippet to make sure that the `instrOps` array is equivalent to discriminated unions of the opcodes of the instructions implemented by the extension:

```javascript
const instrOps = [...] as const;
// This implements a type equality check for the above array, providing some static safety
type CheckLE = (typeof instrOps)[number] extends (Instruction["op"]) ? any : never;
type CheckGE = (Instruction["op"]) extends (typeof instrOps)[number] ? any : never;
let _: [CheckLE, CheckGE] = [0, 0];
```

Because TypeScript has conditional types and because constant arrays can be transformed to discriminated union types, if the assignment of `[0,0]` to a type `[CheckLE, CheckGE]` does not throw a type error then the `instrOps` array contains exactly the types in the discriminated union `Instruction["op"]`.

## Final Thoughts

TypeScript actually turned out to be a very good language to develop this framework in. The ability to decide how much type safety I want greatly simplified early development of the framework and allowed me to add type safety via generics in an incremental way. More generally, I think this project is a good prototype for building composable interpreter extensions for Bril. The biggest issue right now is that the use of generics makes some boilerplate code quite verbose and can lead to confusing type errors because of the large number of generics and the dependencies between them. The lack of a composable parser does still prevent this approach from being integrated into the codebase as is but is an interesting step in that direction.

The code for this project can be found [here](https://github.com/Dan12/bril/tree/extensible)
