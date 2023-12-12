+++
title = "Variable Arguments in Bril and Arrays in the TypeScript compiler"
[extra]
bio = """
 Alice is an M.Eng student studying Computer Science. She is broadly interested in compilers, systems and algorithm design.
"""
[[extra.authors]]
name = "Alice Sze"
+++

# What was the goal?

The overall goal of my project is to improve the capabilities and the user experience of Bril. In particular, I set out to allow the `main` function in Bril to take a variable number of arguments (much like `argc` and `argv` in C/C++), and add support for arrays in the TypeScript compiler to Bril. I thought of adding variable arguments because we currently have to pack a fixed number of values into an array when writing Bril programs, which is not only extra effort but also limits us to testing a very specific set of inputs, e.g. an array of six elements. While it is not a huge modification, it does limit our ability to test our programs and to see the effect of our optimizations on larger inputs. I also included arrays in the TypeScript compiler because quite a few benchmarks are written using it, and I recall someone asking if arrays are supported for the first assignment. I hope that these extensions will make it ever so slightly easier for CS6120ers to write interesting programs in the future.

# What did you do? 
### Variable Arguments
For the first extension, I modified the TypeScript compiler in `brili.ts`. I changed the function `parseMainArguments` to check if `main` takes an integer and a pointer as arguments, in that order. The pointer can be an `int`, `float`, `bool` or `char` pointer, and there is no restriction on the names of the arguments. If it does, the interpreter allocates an array for the command-line arguments, parses them according to the annotated type and stores them into the array. The name of the int argument will be bound to the size of the array and name of the pointer argument will be bound to the pointer to the array in the environment. The user can then use the two arguments like any other variables. When the program terminates, the interpreter frees the argument array since it is the one that allocated it.

```
# ARGS: 97 108 98 101 114 116
@main(size: int, array: ptr<int>) {
  k: int = const 4;
  zero: int = const 0;
  five: int = const 5;
  output: int = call @quickselect array zero five k;
  print output;
}
```

### Arrays in the TypeScript compiler

For this part, I modified the compiler in `ts2bril.ts`. First, I had to change `tsTypeToBril` so that array types are recognized in Bril. I used the TS typechecker to check if the given type is an array type. If so, we map it to the corresponding pointer type in Bril based on what the element type is in Bril.

Then, I changed `emitExpr` to be able to handle array operations. I started by adding a case in the switch statement for array literals, e.g. `{1, 2, 3}`. We evaluate the Bril type of the array, allocate an array in Bril and iterate over the array to store the elements. Next, I added a case for array indexing. To do this, we obtain Bril expressions for the array and the index by calling `emitExpr`, then create a pointer to the indexed element using `ptradd`. Indices are `number`s in TypeScript, but integers in Bril, which is a problem. The implementation currently accepts any `bigint`s as indices, but only `number` literals can be indices. So a programmer can either use `numbers[0n]` or `numbers[0]`, but `numbers[i]` only works if `i` has type `bigint`. The reason for this is that we can try to parse numeric literals as integers, but there is no narrowing operation for variables in Bril currently, and I am not sure if it would make sense for there to be one. This could be extended in the future.

I also changed the existing assignment case to allow assignment to an element in the array. This is similar to evaluating indexed array as an rvalue, mentioned above.

It is the programmer's responsibility to free the arrays, unlike in real TypeScript, because Bril garbage management is manual by default. I overloaded the existing `free` function in `mem.d.ts` so that the programmer can free the array in TypeScript.
```
export function free<T>(array: T): void;
```

The code below 
```
const numbers: bigint[] = [18n,26n];
console.log(numbers[0n])
free<bigint[]>(numbers);
```
gets compiled to 
```
@main {
  v0: int = const 2;
  v1: ptr<int> = alloc v0;
  cur: ptr<int> = id v1;
  v2: int = const 1;
  v3: int = const 18;
  store cur v3;
  cur: ptr<int> = ptradd cur v2;
  v4: int = const 26;
  store cur v4;
  numbers: ptr<int> = id v1;
  v5: ptr<int> = id numbers;
  v6: int = const 0;
  v7: ptr<int> = ptradd v5 v6;
  v8: int = load v7;
  print v8;
  v9: int = const 0;
  v10: ptr<int> = id numbers;
  free v10;
  v11: int = const 0;
}
```

As you can see, there is quite a bit of redundant and dead code due to the way expressions are evaluated in the compiler, but it should be easily eliminated with optimizations.


# What were the hardest parts to get right?
The compiler extension was much trickier than the variable argument one. For the longest time, I did not realize that the link to the declaration file `typescript.d.ts` was broken, because the link to the rest of the code was not. This meant that the compiler was mostly working without it except in a few places, and when I added arrays I thought I must have done something wrong. I wasn't really familiar with the setup of TypeScript and there is very little documentation and few StackOverflow posts on the public TS compiler API, so I was stuck on that part for awhile. It also took some time to find the correct `functions` or `SyntaxKind` for my use cases, as I had to comb through the TypeScript codebase or rely on print statements. For anyone who is going to work with the TypeScript AST, I highly recommend the [AST viewer](https://ts-ast-viewer.com/#).  

Although it was not intuitive at the beginning to use the TypeScript compiler, I feel that it is my biggest takeaway from this project as it allowed me to get some hands-on experience with a "real" compiler that is much more complex than what I built in CS 4120.


# Were you successful? 

For the variable arguments assignment, I rewrote the benchmarks from the `mem` folder where adaptable to test the extension. The benchmarks that I did not rewrite either generate large randomized arrays or used arrays for memoization. I verified that the output from the six rewritten benchmarks matched the original programs with the same arguments to `main`.

For the compiler extension, I wrote programs in TypeScript using arrays, literals and `number`/`bigint` as indices, assignment to arrays etc, iterating over the arrays using for-loops etc (see below). I then ran the compiled Bril programs with the Bril interpreter to verify the outputs.

```
for (let i = 0n; i < 2n;) {
    console.log(numbers[i])
    i = i + 1n;
}
```

Overall, I have enjoyed implementing these features and tinkering with the tools I have been using all semester :) I have also developed a greater appreciation for those who worked on them. 