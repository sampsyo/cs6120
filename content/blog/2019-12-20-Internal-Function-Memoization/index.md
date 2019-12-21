# Internal Function Memoization
Transforming a subset of recursive programs into their iterative counterparts. 

## Motivation
Execution time of a program is important. As a metric, execution time is relevant not just for saving loads of money from uneccessary compute, but is also relevant in terms of functionality. At the limit, writing faster programs can enable us to solve larger problems that require massive computational work in a tangible amount of time.
Compilers can make a given program run faster. Often, we tend to look at classic compiler optimizations as a way to iteratively speed up a given program.

However, the scope of the optimization pass model is limited, especially when the program is algorithmically inefficient. If we are able to transform programs with unneccessarily $O(n)$ space cost into $O(1)$ space cost, or programs with $O(2^n)$ runtime into programs with $O(n)$ runtime, why couldn't a compiler?

One of the challenges of this idea is the notion of an _algorithm_. As humans, we can read the program, determine the task, and write a better program given the necessary computation. However, most compilers do not have a notion of the "goal" of a program. Similar questions arise in program synthesis when determining the "specification" of a program.

## Approach
In order to scope our project to tangible action items, we separated our idea into smaller milestones.

*V0.* Transform a recursive fibonacci program into an iterative program

*V1.* Transform a recursive program with an arbitrary combination of elements on return (ie fibonacci was f(n-1) + f(n-2))

*V2.* Expand scope of input set of valid programs to two-dimensional DP programs

## A First Program

Before digging in to any code, we spent time thinking deeply about recursive DP programs and their iterative counterparts. 
1. What are the disinguishing parts of the recursive implementation that can give rise to an equivalent iterative program?
2. Which parts of these recursive functions is the similar across them such that we can generalize? 
3. How do we know these are equivalent?

We decided to implement this at the LLVM-IR level so that our optimization would be source-language-independent.

It is common knowledge that there are [hard problems](https://en.wikipedia.org/wiki/List_of_unsolved_problems_in_mathematics) in the field of mathematics. Therefore, we avoid mathematics and endeavour on an ad-hoc approach. This also results in a more reader-friendly blogpost. 

We came up with several hypothesis that we then used to develop a model for _V0_.

### Hypotheses

##### Hypothesis: Memoized programs only need to store a number of values equal to the number of recursive calls.

##### Hypothesis: Memoized programs need to iterate on a monotonically incrementing value.

##### Hypothesis: Memoized programs and agnostic to the actual computation being done.

### Information Collection
Given a recursive, what information do we need to be able to generate a iterative version?
- Base Cases
- Loop induction variable and upper bound
- How these recursive calls are combined at return (i.e., fibonacci adds the results from the f(n-1) and f(n-2) together)
- The offset from our induction variable in the recurisve calls (i.e., (n-1), (n-3), etc.)

Information collection posed many challenges due to the low-level of LLVMIR, and desire to generalize our solution.
#### Finding the Base Cases
To more formally define the term _base cases_, we required base cases to be comparisons with the input argument to the function, found at the top of a program, that resulted in returning a value within the body guarded by the comparison.
While humans have no trouble decipering the base case given a program, it is much more challenging to find them when sifting through LLVMIR. We used an ad-hoc approach where we found the constant return values and then walked up the CFG to find the conditional statements that would have led to those constant outputs. From the condition instruction we extracted the function argument that would have led to that constant to be returned, and we end up with a list of argument-return pairs, which sufficiently describes the base cases of the function.

Doing this better would involve abstract interpretation. For a particular function argument, it could trigger a recursive call or it could not. With abstract interpretation, we could find all values of the argument such that the basic blocks with recursive function calls are not traversed. Those would be the base cases. 


### Creating a Model for Iterative Programs to use Collected Information
Once we have the information from the recursive program, we need to find a model such that given this information, can generate a iterative program. To determine what output we wanted, we wrote a iterative fibonacci program. Instead of writing a single transformation funtion $f$(Input Recursicve Progam) = Output Iterative Program, we break down the programs according to a layout. 

Consider the layout of recursive and iterative fibonacci programs below.
![](https://i.imgur.com/tXPAkU2.png)
Here, we separate the base cases from the recursive calls in order to know which set of instructions will need to be in a loop, and which set will only need to be called once. It is important to note that in the black box, you can have any set of instructions to combine the recurisve calls.

Consider the layout of the iterative version of the same program below.
![](https://i.imgur.com/LksHz3E.png)
Note: We keep the function header the same in order to more easily swap out invocations to fibonacci outside the function with our generated function.

### New Code Generation
To minimize changes to the overall code struture, we delete all blocks and instructions inside the original recursive function and insert our new instructions.

First we add all the base case conditionals. Next, we decalre and initialize all the values we need for iteration and the incrementing iterator. After that, we create the while loop and clone in the instructions dependent on the original recursive calls (e.g., the add instruction in fib(n-1) + fib(n-2)). Lastly, we add the return statementment to which the while loop exits. 
## Program Requirements and Extensibility

![](https://i.imgur.com/uCtIcSt.png)

## Evaluation
### Correctness
In order to evaluate correctness, we look at the programs generated for the set of valid benchmarks. These programs include programs listed below
### Performance
To measure performance, we run derivations of the fibonacci program. We compare the execution time of the recursive program to the execution time of the iterative program.

Here are the results of running the fibonacci benchmark shown above on 7 and 34. When computing the fibonacci number 7, our generated program was 1.23x faster than the recursive program. Furthermore, computing the 34th fibonacci number was 1.78x faster.
![](https://i.imgur.com/YD05Swy.png)

When interpreting these results, we considered the difference between these two programs. 
Consider the call tree for recursive fibonacci. As $n \longrightarrow \infty$, the number of calls in the call tree grow by $O(2^n)$.
![](https://i.imgur.com/xaaXLDV.png)
In addition to the redundant computation, the recursive fibonacci progarm takes up more space, as it is not tail-recursive, and therefore requires many more stack frames for each computation.

Next consider Katinacci, a derivation of fibonacci with the recurrence: $$k(n) = k(n-1) + k(n-3)$$
![](https://i.imgur.com/imheTfn.png)



Katinacci was designed to test recurrences with offsets that were not adjacent. 

Finally, Henrinacci is a derivation of fibonacci with the recurrence: $$h(n) = h(n-1) + h(n-2) + h(n-3)$$
![](https://i.imgur.com/k65lo6w.png)
We see similar results to Katinacci for the 7th Henrinacci number, but see quite larger numbers for the 34th and 40th numbers in our sequence. This is likely due to integer overflow, as we only used 32-bit integer values in our LLVMIR. Therefore, this data was less informative to plot.