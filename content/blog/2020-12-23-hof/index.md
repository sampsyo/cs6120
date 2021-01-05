+++
title = """Higher-Order Functions in Bril"""
[extra]
bio = """
Priya Srikumar is a senior undergraduate at Cornell University. They like researching compilers, verification, and programming languages!

Goktug Saatcioglu is a first-year PhD student of computer science at Cornell University who is interested in programming languages, security and distributed systems.
"""
[[extra.authors]]
name = "Priya Srikumar"
link = "https://priyasrikumar.github.io"
[[extra.authors]]
name= "Goktug Saatcioglu"
+++

## What was the goal? 

We were aiming to implement higher-order functions (HOFs) in Bril by implementing a lambda lifter (which makes higher-order functions independently defined from each other globally) and adding support for local function definitions into the Bril syntax. We were also hoping to have functions as function arguments and return types, but this turned out to be more difficult than anticipated.

## What did we do? 

Everything we set out to do except for functions as function arguments and returning functions. 

### Design

Bril is an instruction-based language, so it was tricky to think about what functions would look like. We decided that higher-order functions would be defined with an `anon` keyword, and that they would be applied with the keyword `apply`.

#### Types

We created a new parametrized `fun` type with the following syntax: 

```
{
    "fun": {
        "params" : [<Type>*]?, 
        "ret" : [<Type>]?
    }
}
```
Both the `params` and the `ret` field are optional. 

#### Operations 

`anon` : Indicates a function definition. The first argument is a comma-separated parenthesis list of argument names. The second argument is a bracketed list of instructions and labels (like a normal function definition). 

`apply` : Indicates a function application. The first argument is the function name. The subsequent arguments are arguments to that function.

Here's an example of those two operations being used in a Bril program (in text format for readability): 

```
@main {
    a : int = const 1; 
    b : int = const 2;
    add : fun<int, int>, <int> = anon(x, y) {
            res : int = x + y; 
            ret res;
        };
    res : int = apply add a b;
    print res;
}
```

### Implementation 

#### Intuition 

Lambda lifting as an idea is pretty straightforward. First, make sure all the functions are uniquely named. Then, do the following until there are no more free variables or local functions: Replace all free variables to the higher-order function with an extra argument to the outer function that gets passed in to the higher-order one. Then, pull out every local function definition without free variables into the global scope. 

Concretely, a lifted version of the Bril program above would look like this:

```
@add(x: int, y: int) : int {
    res : int = x + y; 
    ret res;
}

@main {
    a : int = const 1; 
    b : int = const 2;
    res : int = call @add a b;
    print res;
}
```

#### Code Breakdown

We had three main functions: `free_vars`, which took in a function's arguments and body and determined the free variables it contained, `process_func`, which took in a function's arguments as well as its basic blocks and returned a list of extracted global functions as well as the original function's new block structure, and `process_q`, which took in a program's CFG and queued each of its functions for processing with `process_func`, adding on new global functions to extract other functions from to the queue. 

#### `bril-txt` modification 

We updated `briltxt.py` to translate Bril programs using HOFs in a textual representation into the JSON representation (and vice versa). This made it a lot easier to write and run test cases! 

## What was hard to get right? 

A saying (by Phil Karlton, allegedly) that gets riffed on a lot among computer scientists goes: *There are 2 hard problems in computer science: cache invalidation, naming things, and off-by-1 errors*. We encountered all of the above when implementing HOFs in Bril. 

Function renaming was tricky as the mangling scheme we implemented made it hard to keep track of which function names mapped to each function in our cache of functions. We ended up scrapping the mangling scheme as it was never a problem in the example programs and we figured we could require unique names as part of the specification. We also had some issues with queueing extracted global functions into our lambda lifter; we were always missing the last one. We're not sure if this was because we were working in OCaml, where mutability is not welcomed, but after making the code more explicitly imperative this problem vanished. 

We discovered that we couldn't take functions as arguments or return them since that would create a closure, which is a more general form of the nested functions we were implementing. To implement the desired functionality, we would need to develop a closure semantics, which would require a way to express environment values. We think that implementing structs in Bril could be a good way to store a function's state in order to do this. Another related approach would be to use functions pointers that can get passed around as arguments and returned ([LLVM does this][fp]). However, this was beyond what we were able to accomplish this semster. 

## Were we successful? 

We went through the [Bril benchmarks][bench] and selected the ones that had multiple functions. The Bril benchmarks have been created by students that have taken the class and include programs that multiply matrices, compute orders of cyclic groups, and calculate the Ackermann function, among others. We modified these programs to have multiple layers of nesting and/or multiple nested functions in a single function. These programs can be found in [our repository][repo] under `test/hof/`. We then lifted these modified programs and compared the number of dynamic instructions we got to the number that interpreting the original programs directly yielded; these results can be found in `out/hof.csv`. In most cases, we had the same number of instructions, which is at least some indicator that we were doing something right methodologically. We noticed that for programs where we had multiple nested functions and/or multiple nested function calls, the number of instructions increased by 1 or 2. `binary-fmt.bril` was a notable exception to this: it had 10 extra instructions. We suspect that this was due to the recursive nature of one of the functions as well as the calls to another HOF inside the recursive body. 

## Future Directions

We would actually like to implement closures in Bril before the next iteration of this course. This will involve tweaking the way we analyze function arguments, but we haven't figured out the specifics yet. 

[bench]:https://github.com/sampsyo/bril/tree/master/benchmarks
[fp]:https://mapping-high-level-constructs-to-llvm-ir.readthedocs.io/en/latest/basic-constructs/functions.html
[repo]:https://github.com/priyasrikumar/6120