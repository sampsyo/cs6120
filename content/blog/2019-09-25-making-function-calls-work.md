+++
title = "Bril()"
extra.author = "Gautam Mekkat, Eashan Garg, and Sameer Lal"
extra.bio = """
"""
+++

## Bril()

### Goal
Our goal for this project was to make function calls work. Functions reduce code duplication and allow programmers to abstract away common functionality. This simplifies the development and improves the readability of complex programs. To make functions even more useful, we introduce function parameters, return types, optional type annotations for parameters, nested function definitions, and a simple module system. We also offer the option to pass command-line arguments to the main function.

### Design

#### Adding support for function calls
The provided interpreter already supports the evaluation of functions. All we have to do is add a *call* instruction to the language grammar and link that to the interpreter. To limit function scope, we create a new environment (a map from variable names to values) every time a function is called.

Below is a Bril program that demonstrates this functionality.

    # This program prints out 100 and exits. Nothing too exciting here.
    main {
      call func;
    }
    
    func {
      v0: int = const 100;
      print v0;
    }

Since the interpreter scans function definitions before executing anything, functions can be defined in any order. The above program can be rewritten as:

    func {
      v0: int = const 100;
      print v0;
    }

    main {
      call func;
    }


#### Adding function parameters
To add support for function parameters, we first update the grammars for function definitions and calls to take whitespace-delimited lists of variable names. The interpreter's `call` operation handler first extracts the values for all arguments from the current environment map. It then pre-populates the callee's environment with the function parameters mapped to these values. This new environment is then used when evaluating the called function.

Below is a Bril program that demonstrates the use of function parameters.

    # This program also prints out 100 and exits.
    main {
      call print_double 50;
    }
    
    print_double x {
      v0: int = add x x;
      print v0;
    }

#### Return types
Without the ability to return data, our functions are not very useful. To add support for return types, we first update the grammar for function definitions to optionally take `: type` at the end of the header, and update the rule for `ret` to optionally take a variable name to return. We also overload the `call` operation to be both an effect operation and a value operation, since functions that do not return anything will be effect operations and those that do will be value operations.

To pass a return value back to the caller, we use the following approach. When we handle a `ret` operation, if there is a return value, we add a special variable `_ ret` to the environment and map it to the return value. Note that this will never collide with any existing variables as there is space in the name. Then, if the caller is expecting a return value (which it will in instructions of the form `v: type = call func_name`), it can check for the existence of `_ ret` in the callee's environment. We do some basic type-checking here, comparing return types and variable types, to make the Bril programmer's life easier.

Below is an example program that demonstrates this functionality.

    # Once again, we print 100.
    main {
      v0: int = call get_hundred;
      print v0;
    }
    
    get_hundred: int {  # The ": int" is required.
      v0: int = const 100;
      ret v0;
    }

Since we now have the ability to pass arguments to functions and get return values, we can now write some interesting Bril programs. Below is a Bril program that prints the 10th Fibonacci number.

    # Print fib(10)
    main {
        v0: int = const 10;
        fib10: int = call fib v0;
        print fib10;
    }

    # Return true if n <= 1, false otherwise.
    lte_one n: bool {
        one: int = const 1;
        lto: bool = le n one;
        ret lto;
    }

    # Return fib(10).
    fib n: int {
        base: bool = call lte_one n;
        br base return continue;
    return:
        ret n;
    continue:
        one: int = const 1;
        prev: int = sub n one;
        prev2: int = sub prev one;
        fib1: int = call fib prev;
        fib2: int = call fib prev2;
        ans: int = add fib1 fib2;
        ret ans;
    }

**Note**: To call a function on the right-hand side of an assignment, the function must declare a return type. We choose to enforce this to improve readability. The following program will fail with the error message `function func does not return`.

    # Fails!
    main {
      v0: int = call func;
      print v0;
    }

    func {
      v0: int = const 100;
      ret v0;
    }

The opposite is allowed: functions that have a declared return type *may* be called as standalone instructions. The following program succeeds and prints 100.

    # Print 100
    main {
      call func;
    }

    func: int {
      v0: int = const 100;
      print v0;
      ret v0;
    }


#### Optional type annotations for function parameters
Function definitions are now more complex, and with this added syntactic complexity comes a loss of readability. To fix this, we introduce the notion of type annotations for function parameters. Since we do not do support static type checking, these type annotations serve no operational purpose. As a result, all we have to do is update the language grammar. Below is an updated version of the above program with type annotations.

    # Print fib(10)
    main {
        v0: int = const 10;
        fib10: int = call fib v0;
        print fib10;
    }

    # Return true if n <= 1, false otherwise.
    lte_one (n: int): bool {
        one: int = const 1;
        lto: bool = le n one;
        ret lto;
    }

    # Return fib(10). Requires: n >= 0.
    fib (n: int): int {
        base: bool = call lte_one n;
        br base return continue;
    return:
        ret n;
    continue:
        one: int = const 1;
        prev: int = sub n one;
        prev2: int = sub prev one;
        fib1: int = call fib prev;
        fib2: int = call fib prev2;
        ans: int = add fib1 fib2;
        ret ans;
    }

#### Nested function definitions
In the above example, `lte_one` is just a helper function for `fib`and is not used anywhere else. To avoid cluttering the global function definitions, it would be nice to only define functions where they are useful.

To do this, we introduced support for nested function definitions (i.e., function definitions within function definitions). First we add a new rule to the grammar. We add a new `instr` rule of the format `"def" func`, where `func` is the rule for normal function definitions. The `"def"` is there to avoid issues with labels (when the parser encounters `x:` it will not know if `x` is a label or a function with a return type). In the interpreter, we introduce the notion of a local function map. Previously, we had a global function map for the entire program. The local function map restricts access to a nested function to the immediate parent function.

We also change the `call` operation handler to first search the local map before searching the global map. Note that this means functions can be shadowed. Below is the above program updated with nested function definitions.

    # Print fib(10)
    main {
        v0: int = const 10;
        fib10: int = call fib v0;
        print fib10;
    }

    # Return fib(10). Requires: n >= 0.
    fib (n: int): int {
        def lte_one (n: int): bool {
            one: int = const 1;
            lto: bool = le n one;
            ret lto;
        }
        base: bool = call lte_one n;
        br base return continue;
    return:
        ret n;
    continue:
        one: int = const 1;
        prev: int = sub n one;
        prev2: int = sub prev one;
        fib1: int = call fib prev;
        fib2: int = call fib prev2;
        ans: int = add fib1 fib2;
        ret ans;
    }

#### Command-line arguments
Adding support for command-line arguments is quite straightforward. We can use the `process.argv` variable to access arguments to `brili`. For each argument, we simply ensure that the argument is either an integer or a string containing "true" or "false", in which case we convert the string to the corresponding Boolean. We then pass these arguments to `main` the same way we would pass arguments to any other function.

We can update the above Fibonacci implementation to take in a command-line argument `n`, and print out `fib(n)`.

    # Print fib(n)
    main n {
        fibn: int = call fib n;
        print fibn;
    }

    # Return fib(10). Requires: n >= 0.
    fib (n: int): int {
        def lte_one (n: int): bool {
            one: int = const 1;
            lto: bool = le n one;
            ret lto;
        }
        base: bool = call lte_one n;
        br base return continue;
    return:
        ret n;
    continue:
        one: int = const 1;
        prev: int = sub n one;
        prev2: int = sub prev one;
        fib1: int = call fib prev;
        fib2: int = call fib prev2;
        ans: int = add fib1 fib2;
        ret ans;
    }

If the above program were in a file called `fib.bril`, we could run it by running:
```bril2json < fib.bril | brili [n]```

#### Module system
Lastly, we introduce a basic module system to allow for basic abstraction of functionality. The syntax for importing modules is `import MOD_NAME;`. All imports must be placed at the top of the Bril file, and `MOD_NAME.bril` should exist in the current working directory. After parsing the Bril file, in addition to returning the list of functions, we include a list of imported module names (we don't do anything else with the module names; this will become clear shortly). We added a new command in addition to `bril2json`and `ts2bril`, named `loadbril`. This command takes the new IR representation which includes the list of imported modules and recursively parses and loads each module. Circular imports *are* supported, as this is a very basic module system that simply adds all imported functions to the global namespace. (We considered adding namespaces for modules but decided that this would best be left for a separate project.) We also detect duplicate function definitions as the module system could make detecting these manually much more tedious.

---

### Evaluation
By adding the above features to Bril, the language becomes more complex, increasing room for error. We have added many rigorous tests to ensure the correctness of our implementation.

We considered benchmarking the performance of our language features but decided that this did not make much sense, as we did not have a baseline metric. We thought about comparing our language's performance to that of another interpreted language such as Python but decided that there were too many variables to consider and that our results would most likely be misleading.

#### Basic function calls
With basic function calls, we want to make sure that execution jumps to the desired functions. We also want to make sure programs follow standard scoping rules (variables are scoped within their respective functions). Take the following test program:

```
main {
  v0: int = const 4;
  call double v0;
}

double a {
  v0: int = const 2;
  v1: int = mul a v0;
  print v1;
}
```

In the above program, `v0` is defined in `main` and `double`. When `double` is called, `v0` is passed in as `a`, and `double` has no knowledge of any variables in `main`. (Fortunately, the initial Bril interpreter handles basic function scoping.)


#### Function parameters
We evaluate the correctness of function parameters in two stages.

First, we need to ensure that the new function definition grammar is parsed correctly. Each statement accepts a variable number of arguments, each of which can be assigned a type. We test this by defining functions with 0, 1, and 10+ arguments and make sure the parser output is as expected.

Second, we need to ensure that we are able to pass arguments. To test this, we test several functions that take a varying number of arguments and ensure that calling them with different arguments would execute correctly.

#### Return statements
Return statements add another layer of complexity– now we want to pass data out of functions. The testing scheme here is similar to the one for function parameters.

#### Nested functions
For nested functions, the primary testing benchmark is to see if nested functions behave similar to normal functions within scope of the parent. To do this, we test the following cases:
- one function nested in another function,
- two functions nested in one function, 
- one nested function inside a nested function of another function, and
- nested function name shadowing.

We designed nested functions in Bril to obey lexical scope, so our test cases enumerate all possible cases. An example of two Bril programs, one with nested function definitions, and one without, is the above recursive Fibonacci implementation.

#### Modules
The most important aspects of our testing plan for the module system are circular imports and duplicate function definitions across modules. Since our module system is quite simple and circular imports are allowed, we simply test programs that import in a circular fashion. Some of these cases include:
- `a` and `b` import each other,
- `a`, `b`, and `c` all import each other, and
- `a` imports `b`, `b` imports `c`, `c` imports `d`, and `d` imports `a`

To test duplicate function definitions (which our module system should detect and complain about), we test duplicate function definitions across imported modules and make sure that the module loader raises an exception.

#### Command-line arguments
To test command-line arguments, we run programs taking various numbers of arguments of both integer and Boolean types.

---

### Notable challenges
One notable challenge we encountered when implementing our language features was designing the loading system for modules. We ended up on a rather simple design (see above), but tested various other implementations. We first tried loading modules as soon as the `import` statement was encountered. This made it difficult to detect circular imports. We also tried updating the `loadbril` command to take the list of module files, but found this to be detrimental to usability, as running programs that imported many modules would require enumerating the filenames of all the modules. We also considered adding namespaces for modules as discussed in the previous section, but could not settle on a clean design.