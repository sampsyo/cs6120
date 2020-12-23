+++
title = """Higher-Order Functions in Bril"""
[extra]
bio = """
Priya Srikumar is a senior undergraduate at Cornell University. They like researching compilers, verification, and programming languages!"""
[[extra.authors]]
name = "Priya Srikumar"
link = "priyasrikumar.github.io"
+++

## What was the goal? 

I was aiming to implement higher-order functions (HOFs) in Bril by implementing a lambda lifter (which makes higher-order functions independently defined from each other globally) and adding support for local function definitions into the Bril syntax. I was also hoping to have functions as function arguments and return types, but this turned out to be more difficult than anticipated.

## What did I do? 

Everything I set out to do except for functions as function arguments and returning functions. 

### Design

Bril is an instruction-based language, so it was tricky to think about what functions would look like. I decided to allow variables to be instantiated with an `anon`

#### Types

I created a new parametrized `fun` type with the following syntax: 

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

Here's an example of those two operations being used in a Bril program: 

```
a : int = const 1; 
b : int = const 2;
add : fun<int, int>, <int> = anon(x, y) {
        res : int = x + y; 
        ret res;
    };
res : int = apply add a b;
```

### Implementation 

#### Intuition 

Lambda lifting as an idea is pretty straightforward. First, make sure all the functions are uniquely named. Then, do the following until there are no more free variables or local functions: Replace all free variables to the higher-order function with an extra argument to the outer function that gets passed in to the higher-order one. Then, pull out every local function definition without free variables into the global scope.

#### Code Breakdown

I had three main functions: `free_vars`, which took in a function's arguments and body and determined the free variables it contained, `process_func`, which took in a function's arguments as well as its basic blocks and returned a list of extracted global functions as well as the original function's new block structure, and `process_q`, which took in a program's CFG and queued each of its functions for processing with `process_func`, adding on new global functions to extract other functions from to the queue. 

#### `bril-txt` modification 

I updated `briltxt.py` to translate Bril programs using HOFs in a textual representation into the JSON representation (and vice versa). This made it a lot easier to write and run test cases! 

## What was hard to get right? 

A saying (by Phil Karlton, allegedly) that gets riffed on a lot among computer scientists goes: *There are 2 hard problems in computer science: cache invalidation, naming things, and off-by-1 errors*. I encountered all of the above when implementing HOFs in Bril. 

Function renaming was tricky as the mangling scheme I implemented made it hard to keep track of which function names mapped to each function in my cache of functions. I ended up scrapping the mangling scheme as it was never a problem in the example programs and I figured I could require unique names as part of the specification. I also had some issues with queueing extracted global functions into my lambda lifter; I was always missing the last one. I'm not sure if this was because I was working in OCaml where mutability is not welcomed, but after making the code more explicitly imperative this problem vanished. 

I discovered that I couldn't take functions as arguments or return them since that would create a closure, which is a more general form of the nested functions I was implementing. To implement the desired functionality, I would need to develop a closure syntax, which was beyond what I was able to accomplish this semster. 

## Was I successful? 

I went through the Bril benchmarks and selected the ones that had multiple functions. I modified these programs to have multiple layers of nesting and/or multiple nested functions in a single function. These programs can be found in my repository under `test/hof/`. I then lifted these modified programs and compared the number of dynamic instructions I got to the number that interpreting the original programs directly yielded. In most cases, I had the same number of instructions, which is at least some indicator that I was doing something right methodologically. I noticed that for programs where I had multiple nested functions and/or multiple nested function calls, the number of instructions increased by 1 or 2. `binary-fmt.bril` was a notable exception to this: it had 10 extra instructions. I suspect that this was due to the recursive nature of one of the functions as well as the calls to another HOF inside the recursive body. 

## Future Directions

I would actually like to implement closures in Bril before the next iteration of this course. This will involve tweaking the way I analyze function arguments, but I haven't figured out the specifics yet. 