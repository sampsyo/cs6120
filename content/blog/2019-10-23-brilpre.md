+++
title = "BrilPRE: A tool for Partial Redundancy Elimination on Bril"
extra.author = "Siqiu Yao"
extra.bio = """
  [Siqiu Yao](http://www.cs.cornell.edu/~yaosiqiu/) is a 3rd year Ph.D. student interested in security about systems and programming languages.
"""
+++

### Problem
Partial redundancy elimination (PRE) is a compiler optimization that eliminates expressions that 
are redundant on some but not necessarily all paths through a program.

For example, in the following code:
```
main {
    a:int = const 20;
    n:int = const 1000;
    cmp:bool = const true;
    br cmp here end;

    here:
    n:int = add a a;

    end:
    b:int = add a a;
    print n;
}
```

The expression `a + a` assigned to `b` is partially redundant when `cmp` is true.
One possible PRE optimization would be:
```
main {
    a:int = const 20;
    n:int = const 1000;
    cmp:bool = const true;
    br cmp here newbranch;
    
    newbranch:
    tmp:int = add a a;
    jmp end;

    here:
    tmp:int = add a a;
    n:int = id tmp;

    end:
    b:int = id tmp;
    print n;
}
```
Now no matter in which path, `a + a` would be only computed once. 

### Design
This tool applies the algorithm called [lazy code motion](https://dl.acm.org/citation.cfm?id=143136).
While preserving *computational optimality* (no partial redundant expression computations), 
this algorithm guarantees *lifetime optimality* (computations are placed as early as necessary but as late as possible).

This algorithm involves four passes over the control flow graph (CFG) of the program.

Before the passes, this algorithm will add a new empty block between each block with multiple predcessors with its predecessor. And in the end, all empty blocks will be removed. 

#### Pass 1: Anticipated Expressions
An expression is *anticipated* at a point if it is certain to be evaluated 
along any path before defined (variables involved are reassigned).
```
use(b) = set of expressions evaluated before killed in b
def(b) = set of expressions whose related variables are reassigned in b 

anticipated.out(b) = union over anticipated.in(b') for b' in successors of b
anticipated.in(b) = (anticipated.out(b) - def(b)) union use(b)

anticipated.in(exit) = empty set
```

#### Pass 2: Available Expressions
An expression is *available* at a point if it is available in the usual sense 
**assuming all anticipated expressions at this point are precomputed**.
```
kill(b) = set of expressions defined and not evaluated afterward

available.in(b) = intersection over available.out(b') for b' in predecessors of b
available.out(b) = (available.in(b) union anticipated.in(b)) - kill(b)
```
Then for each expression, 
we want to find the blocks where this expression is anticipated but not available at the beginning.
```
earliest(b) = anticipated.in(b) - available.in(b)
``` 
`earliest(b)` intuitively indicates expressions must be evaluated before this block.

#### Pass 3: Postponable Expressions
This step aims to achieve lifetime optimality, that is, 
delay the evaluation of expressions as long as possible.

An expression is *postponable* at a point if for every path arrives this point, 
this expression was in set `earliest` but never used.
```
postponable.in(b) = intersection over postponable.out(b') for b' in successors of b
postponable.out(b) = (postponable.in(b) union earliest(b)) - use(b)
``` 
Then now we can compute points that certain expressions must be evaluated.
```
latest(b) = 
    (earliest(b) union postponable.in(b)) 
        intersect 
    (kill(b) union 
        not(intersection over (earliest(b') union postponable.in(b')) for b' in successors of b))
``` 
`latest(b)` intuitively indicates expressions that can be plated in b and not ok to put at some of the successors.

#### Pass 4:
We can already place evaluations in `latest(b)` for each block `b`, 
this step tries to solve this problem:
when an expression will only be used once after this evaluation, 
there is no need to place this evaluation.

An expression is *used* at a point if it will be used along some path from this point.
```
used.out(b) = union over used.in(b') for b' in successors of b
used.in(b) = (used.out(b) union use(b)) - latest(b)
``` 

Finally, we insert evaluations of expressions in both `used.out(b)` and `latest(b)` 
and replace the latter usages of these expressions.  


### Implementation
[This tool](https://github.com/Neroysq/BrilPRE) is implemented in Java. 
I leveraged some code from my last project, such as the parsing of Bril JSON files.

There are some tricks applied when implementing this algorithm:
1. When building the control flow graph, 
the implementation treats each instruction as one block. 
So during all the passes, we can ensure that each block can only contain one instruction.
This makes analyzing easier.  

2. When inserting new variables and labels, we must make sure the new names are unique,
the way I do is find the smallest `n` where the string `n`*"_" is not a prefix of any
existing variables in the code. 
Then I can create any name and put this prefix on to get rid of any conflict.

3. When storing a value operation, this tool will first normalize the expression
to try to make arguments sorted. If the order of the arguments need to be reversed, 
I will also reverse the operation name (i.e., `le` to `ge`, `add` to `add`).  
Then we can store expressions as strings in later steps.

  
 

### Evaluation
I manually designed a couple of test cases (in folder `pre_test`) and also pulled test cases in
repo `bril`.

To test the correctness, I compare the results run by `brili` between PRE optimization, 
they match perfectly.

I also want to evaluate how good PRE performs. I consider three measurements: 
line of code, instructions executed, and computations executed.

To measure lines of code, I wrote a script in Python to count lines of instructions in source code.

To count instructions executed, 
I hacked the reference interpreter `brili` to count instructions and show it at the end when executing.
However, I found this number not representative: 
Even PRE gets rid of redundant evaluations, it doesn't decrease the number of instructions executed,
because it only replaces original evaluations with `id` operations, not removing them.

Therefore, I only count those computationally significant operations (all value operations except `id`), 
plus `br` and `jmp`.

| testcase                                  | LoC before | LoC after | diff  | #instr before | #instr after | diff  | #comp instr before | #comp instr after | diff   |
|-------------------------------------------|------------|-----------|-------|---------------|--------------|-------|--------------------|-------------------|--------|
| ./test/dom_test/loopcond.json             | 22         | 22        | 0.0%  | 117           | 117          | 0.0%  | 82                 | 82                | 0.0%   |
| ./test/tdce_test/skipped.json             | 6          | 6         | 0.0%  | 4             | 4            | 0.0%  | 1                  | 1                 | 0.0%   |
| ./test/tdce_test/double-pass.json         | 6          | 6         | 0.0%  | 6             | 6            | 0.0%  | 2                  | 2                 | 0.0%   |
| ./test/tdce_test/reassign-dkp.json        | 3          | 3         | 0.0%  | 3             | 3            | 0.0%  | 0                  | 0                 | na     |
| ./test/tdce_test/combo.json               | 6          | 6         | 0.0%  | 6             | 6            | 0.0%  | 2                  | 2                 | 0.0%   |
| ./test/tdce_test/double.json              | 6          | 6         | 0.0%  | 6             | 6            | 0.0%  | 2                  | 2                 | 0.0%   |
| ./test/tdce_test/simple.json              | 5          | 5         | 0.0%  | 5             | 5            | 0.0%  | 1                  | 1                 | 0.0%   |
| ./test/tdce_test/diamond.json             | 11         | 11        | 0.0%  | 6             | 6            | 0.0%  | 2                  | 2                 | 0.0%   |
| ./test/tdce_test/reassign.json            | 3          | 3         | 0.0%  | 3             | 3            | 0.0%  | 0                  | 0                 | na     |
| ./test/lvn_test/redundant.json            | 6          | 7         | 16.7% | 6             | 7            | 16.7% | 3                  | 2                 | -33.3% |
| ./test/lvn_test/idchain.json              | 5          | 5         | 0.0%  | 5             | 5            | 0.0%  | 0                  | 0                 | na     |
| ./test/lvn_test/nonlocal.json             | 8          | 9         | 12.5% | 7             | 8            | 14.3% | 4                  | 3                 | -25.0% |
| ./test/lvn_test/idchain-prop.json         | 5          | 5         | 0.0%  | 5             | 5            | 0.0%  | 0                  | 0                 | na     |
| ./test/lvn_test/idchain-nonlocal.json     | 7          | 7         | 0.0%  | 6             | 6            | 0.0%  | 1                  | 1                 | 0.0%   |
| ./test/lvn_test/commute.json              | 6          | 7         | 16.7% | 6             | 7            | 16.7% | 3                  | 2                 | -33.3% |
| ./test/lvn_test/clobber.json              | 10         | 11        | 10.0% | 10            | 11           | 10.0% | 5                  | 3                 | -40.0% |
| ./test/lvn_test/redundant-dce.json        | 6          | 7         | 16.7% | 6             | 7            | 16.7% | 3                  | 2                 | -33.3% |
| ./test/lvn_test/clobber-fold.json         | 10         | 11        | 10.0% | 10            | 11           | 10.0% | 5                  | 3                 | -40.0% |
| ./test/lvn_test/reassign.json             | 3          | 3         | 0.0%  | 3             | 3            | 0.0%  | 0                  | 0                 | na     |
| ./test/pre_test/complex_loop.json         | 14         | 15        | 7.1%  | 4009          | 4010         | 0.0%  | 4004               | 3005              | -25.0% |
| ./test/pre_test/complex_loop2_unsafe.json | 18         | 18        | 0.0%  | 7852          | 7852         | 0.0%  | 6867               | 6867              | 0.0%   |
| ./test/pre_test/complex_loop2.json        | 19         | 21        | 10.5% | 6007          | 6508         | 8.3%  | 5501               | 5001              | -9.1%  |
| ./test/pre_test/register_presure.json     | 14         | 16        | 14.3% | 9             | 10           | 11.1% | 4                  | 4                 | 0.0%   |
| ./test/pre_test/simple_loop.json          | 9          | 13        | 44.4% | 7             | 8            | 14.3% | 3                  | 2                 | -33.3% |
| ./test/pre_test/logic.json                | 21         | 21        | 0.0%  | 6             | 6            | 0.0%  | 2                  | 2                 | 0.0%   |
| ./test/pre_test/print.json                | 3          | 3         | 0.0%  | 3             | 3            | 0.0%  | 0                  | 0                 | na     |
| ./test/pre_test/add.json                  | 3          | 3         | 0.0%  | 3             | 3            | 0.0%  | 1                  | 1                 | 0.0%   |
| ./test/pre_test/loop_invariant.json       | 11         | 12        | 9.1%  | 400005        | 400006       | 0.0%  | 400000             | 300001            | -25.0% |
| ./test/pre_test/fibonacci.json            | 17         | 17        | 0.0%  | 648           | 648          | 0.0%  | 403                | 403               | 0.0%   |
| ./test/pre_test/complex_loop3.json        | 24         | 26        | 8.3%  | 66004         | 66006        | 0.0%  | 63999              | 53001             | -17.2% |
| ./test/pre_test/gcd.json                  | 16         | 16        | 0.0%  | 141           | 141          | 0.0%  | 92                 | 92                | 0.0%   |
| ./test/pre_test/factorial.json            | 14         | 14        | 0.0%  | 100008        | 100008       | 0.0%  | 100003             | 100003            | 0.0%   |
| ./test/df_test/cond.json                  | 15         | 15        | 0.0%  | 9             | 9            | 0.0%  | 3                  | 3                 | 0.0%   |
| ./test/df_test/fact.json                  | 13         | 13        | 0.0%  | 62            | 62           | 0.0%  | 42                 | 42                | 0.0%   |

Above is a table showing all the result, 
unfortunately for most cases, the improvement is not significant, 
part of the reasons should be that most programs are short and do not involve loops.
But for all the program about loops that I manually designed,
this tool can successfully detect them, generate correct new code,
and provide a significant performance improvement.

Also, PRE will significantly increase code length. 

### Conclusion
In conclusion, I successfully implemented partial redundancy elimination and tested its correctness and performance.
I hope to investigate more of PRE, such as testing it on more practical programs, 
extending it to eliminate injured partial redundancies, and speculative PRE. 
