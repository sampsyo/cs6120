+++
title = "Shrimp: Verifying IRs with Rosette"

[[extra.authors]]
name = "Rachit Nigam"
link = "https://rachitnigam.com"

[[extra.authors]]
name = "Sam Thomas"
link = "https://github.com/sgpthomas"

extra.bio = """
  [Rachit Nigam](https://rachitnigam.com) is a second year PhD student interested
  programming languages & computer architecture. In his free time, he
  [subtweets](https://twitter.com/notypes/status/1170037148290080771) his advisor and [annoys tenured professors](https://twitter.com/natefoster/status/1074401015565291520).

  Sam Thomas is a senior undergraduate student at Cornell. He is applying to Grad Schools for applied PL.
"""
+++


## The Problem

Writing programs is famously hard. Writing program that generate programs
(compilers) is harder still. Compiler verification usually comes in two
flavors: (1) Proving a compiler is correct by construction using a
[proof-assistant][coq], or (2) proving that each compiler pass preserves the
observable semantics of a program by checking the equivalence of the input and
the output programs.

Non-trivial correct by construction compilers have been demonstrated to be viable for
but require several person-years of work to implement, specify[^1],
and prove correct. On the other hand, proving program equivalence automatically
is a [remarkably hard problem](https://en.wikipedia.org/wiki/Turing_completeness)
which forces such verification efforts to somehow bound the space of program
behaviors.

For our project, we implemented a pass verification infrastructure for Bril using the
[Rosette][] framework and verified the correctness of a local value numbering
pass.

## SMT Solving, Briefy

Underyling a lot of automatic proof generation is SMT solving. Satisfiability
Modulo Theories (SMT) is a generalization of the [SAT][] problem that allows us
to augment our logic with various "theories" (naturals, rationals, arrays, etc.)
to prove properties in a domain that we care about. For example, SAT + theory of
integers can be used to solve [Integer Linear Programming][ilp] problems.

Program properties can be verified by first encoding the semantics of your
language as an SMT formula and asking a solver to prove its correctness by finding
a satisfying assignment.

## Rosette

Rosette is a symbolic execution engine for the [Racket][] programming language.
It lets us write normal Racket programs and does the work of automatically lifting them
to perform symbolic computations. This is different than simply having bindings into an SMT
solver where you use code to generate constraints because Rosette gives symbolic meaning
to actual Racket programs.

Consider the following program:

    def add(x):
      return x + 1

In addition to running this program with _concrete_ inputs (like `1`), Rosette
allows us to run it with a _symbolic input_. When computing with symbolic
inputs, Rosette _lifts_ operations like `+` to return symbolic formulas
instead.  So, running this program with the symbolic input `x` would give us 
the symbolic value `x + 1`.

Rosette also lets us ask _verification queries_ using a symbolic inputs.
We can write the following program:

    symbolic x integer?
    verify (forall x. add(x) > x)

Rosette will convert this into an SMT formula and verify its correctness using
a backend solver.

If we give Rosette a falsifiable formula:

    symbolic x integer?
    verify (forall x. add(x) < x)

Rosette generate a _model_ where the formula is false. In this case, Rosette
will report that when `x = 0`, this formula is false.

## Symbolic Interpretation
A symbolic interpreter is simply an interpreter that executes over symbolic values rather than real values.
A standard interpreter takes an expression, such as `x + 2 + 3`, and a concrete variable assignment, like `x = 1`,
and then recursively evaluates the expression, substituting the value for `x` every time we see it. In this
case `x + 2 + 3` evaluates to `6`. A symbolic interpreter works on the same types of programs, 
but takes symbols as arguments instead of concrete value assignments. For the same program, `x + 2 + 3`, symbolic
interpretation produces the formula `x + 5`. Computations that don't involve symbols are still run concretely and
Rosette is smart enough to do this regardless of the parenthesization of the expression. 

This proves useful for verification because it reduces the problem of program equivalence to formula equivalence.
To prove that the program `x + 2 + 3` is equivalent to the program `3 + 2 + x` we only need to reduce these
to formulas and then prove their equivalence. This still looks hard, but it turns out that we can use SMT
solvers to do most of the hard work.

We have reduced the problem of program equivalence to symbolic interpretation plus a query
to an SMT solver. Fortunately, Rosette makes both of these tasks simple. We can write a normal interpreter for Bril
in Racket and Rosette will lift the computation into SMT formulas and also make the query to the SMT solver.

### Limiting scope to basic blocks

SMT theories are undecidable in general and even when you restrict it to
decidable fragments, verification can take a very long time. Because symbolic
interpretation involves following every path in a program and the number of
paths in a program increases exponentially with the size of the program,
it can be difficult to make verification with symbolic interpretation scale
to large programs.

We address this problem by proving basic block equivalence rather than program equivalence.
By definition, there is only a single path through a basic block. This avoids the exponential
growth of the number of paths to explore and means that we only ever produce relatively simple
formulas that are usually fast to verify. However, this comes at the cost of exact program
equivalence, we can only give a conservative approximation.

To verify that two basic blocks are equivalent, we assume that the common set of live
variables are equal, and ask Rosette to verify that the symbolic formulas we get from interpretation for each
assigned variable are equivalent.


    block1 {
      ...
      sum1: int = add a b;
      sum2: int = add a b;
      prod: int = mul sum1 sum2;
    }

A simple CSE and dead code elimination produces the following code:

    block2 {
      ...
      sum1: int = add a b;
      prod: int = mul sum1 sum1;
    }

We first find the common set of live variables.
In this case, `a, b` are live at the beginning of both of these blocks. Next, we create a symbolic version
of these variables for each block. We'll use `$` to designate symbolic variables.
This gives us `a$1, b$1` for the first block and `a$2, b$2` for the second block. We assume that
`a$1 = a$2` and `b$1 = b$2`. Then we can call our basic block symbolic interpreter with these
variables to get the following formula:

    block1
    sum1 = a$1 + b$1
    prod = (a$1 + b$1) * (a$1 + b$1)

    block2
    sum1 = a$2 + b$2
    sum2 = a$2 + b$2
    prod = (a$2 + b$2) * (a$2 + b$2)

Finally we check if the variables which are defined in both blocks are equivalent.
In other words, assuming that the common live variables are equal, is the following true:

    forall a$1, a$2, b$1, b$2.
    ((a$1 + b$1) = (a$2 + b$2) &&
    (a$1 + b$1) * (a$1 + b$1) = (a$2 + b$2) * (a$2 + b$2))

The SMT solver will verify this for us, and if it can't prove the formula to be valid,
it will provide a counter-example to prove it. In this case, it is not too hard to see
that this formula is in fact valid, which shows that these two basic blocks are functionally
equivalent.

### Downsides
The downside of this approach is that it only conservatively approximates the result
of each basic block. We may lose information about constraints on variables that cross
basic block boundaries. For example, consider the following toy program:

    main {
      a: int = const 2;
      b: int = const 4;
      c: int = id a;
      jmp next;
    next:
      sum: int = add a c;
    }

Because `c` is a copy of `a`, this program would be functionally the same if you replaced the assignment
to `sum` with `sum: int = add a a`. However, because we are only doing verification on the basic block level,
we don't know that these programs are equivalent.

Another problem is that this approach to verification relies on the existence of test programs. We are not
actually analyzing the code of the optimization so if you don't have extensive enough tests, bugs may go by
unnoticed. Of course, you could run this after every invocation of the compiler to increase the likelihood of
finding bugs.

## Evaluation
To evaluate Shrimp, we implemented [Common sub-expression elimination (CSE)][cse]
using [Local value numbering (LVN)][lvn] to show that Shrimp is useful in finding
correctness bugs. We intentionally planted two bugs and found a third bug in the process of testing.

There are some subtleties to a correct implementation of LVN. If you know that the variable
`sum1` holds the value `a + b`, you have to make sure that `sum1` is not assigned to again before
you use it. For example, consider the following Bril program:

    sum1: int = add a b;
    sum1: int = id a;
    sum2: int = add a b;
    prod: int = mul sum1 sum2;

We would like to replace `sum2: int = add a b` with `sum2: int = id sum1` because we
have already computed the value. However, we can't do this directly because then `sum2` would
have the value `a`, not `a + b`. The solution is to rename the first instance of `sum1` to something unique so that we don't lose our reference to the value `a + b`. We can
then replace `sum2` with a copy from this new variable.

We implemented the faulty version and ran Shrimp. It was able to show that the programs
are not equivalent and even produced a counter example to prove this.
With this information, it is easy to walk through the execution of the code
and discover the source of the bug.

Next we tried extending CSE to deal with associativity.
It would be nice if the compiler knew that `a + b` is equal to `b + a` so that it could eliminate more
sub-expressions. The most naïve thing to do is sort the arguments for all expressions when you
compare them so that `a + b` is the same value as `b + a`. However, this by itself is not enough.
Testing the following example with Shrimp reveals the problem:

     sub1: int = sub a b;
     sub2: int = sub b a;
     prod: int = mul sub1 sub2;

Shrimp gives us the counter example `a = -8, b = -4`. The problem is that we can't
sort the arguments for every instruction; $a - b \neq b - a$. Shrimp helps to reveal
this problem.

The final bug was actually an unintentional bug that Shrimp helped us find. We made the arguably
bad decision to give each Bril instruction its own structure that is a sub-type of a `dest-instr` structure
rather than to give `dest-instr` an op-code field. When we were looking up values in the LVN table,
we were only comparing that fields in `dest-instr` were the same. We forgot to compare the actual
types of the instructions! Shrimp was able to reveal this code from the following example:

    sub1: int = sub a b;
    sub1: int = add a b;
    sub2: int = sub b a;
    prod: int = mul sub1 sub2;

This made it easy to find and fix a rather embarrassing bug in the LVN implementation.

## Conclusion

Symbolic verification provides a trade-off between verification effort and
the completeness of a verification procedure. Beyond our implementation,
there has also been recent work in verifying correctness of [file systems][],
[memory models][], and [operating systems][] code using symbolic verification
demonstrating the flexibility of this approach to program verification.


[^1]: The problem of specifying the correctness condition of a compiler is itself
a non-trivial, open research problem. Should the compiler preserve the stdout
behavior, or should it give even stronger guarantees such as preserving the
timing behavior?

[^2]: https://en.wikipedia.org/wiki/Symbolic_execution#Limitations

[rosette]: https://emina.github.io/rosette/
[racket]: https://racket-lang.org/
[coq]: https://coq.inria.fr/
[cse]: https://en.wikipedia.org/wiki/Common_subexpression_elimination
[lvn]: https://en.wikipedia.org/wiki/Value_numbering#Local_value_numbering
[sat]: https://en.wikipedia.org/wiki/Boolean_satisfiability_problem
[ilp]: https://en.wikipedia.org/wiki/Integer_programming
[file systems]: https://homes.cs.washington.edu/~emina/doc/yggdrasil.osdi16.pdf
[memory models]: https://homes.cs.washington.edu/~emina/doc/memsynth.pldi17.pdf
[operating systems]: https://unsat.cs.washington.edu/papers/nelson-serval.pdf
