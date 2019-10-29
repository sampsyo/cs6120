+++
title = "One VM to Rule Them All"

[extra]
bio = """
  [Sameer Lal](https://github.com/sameerlal) is a Master of Engineering student.  He studied Electrical & Computer Engineering and Math in his undergrad.  He is interested in probability-related fields.
"""
[[extra.authors]]
name = "Sameer Lal"
link = "https://linkedin.com/sameer-lal"
+++




# A Short Story
You are the latest hire for the startup company _QuantBet_ which specializes in developing computational models which are then used in proprietary sports betting.  _QuantBet_ is fairly new, and as a gifted PL researcher, you are tasked with creating a new programming language, _QuantPL_ to assist developers in their analysis.

Sports betting is a fast business, so you want your code to run quickly.  You start by developing a parser for _QuantPL_ and Abstract Syntax Tree (AST) interpreter.  But this is too slow, so you write a real VM.  You spend a lot of time on a run-time system in C.

Your company is impressed start using _QuantPL_ which they claim is a more intuitive approach in scripting models.  But soon, they start to notice that it's a lot slower than their previous models.

You now have to design a bytecode format and interpreter, and when they complain even more, you write a JIT compiler.  Now it's fast, and you hear that rival company _Quant2Bet_ has developed a language _Quant2PL_ that's even faster using something called Truffle and Kraal, so now you're stuck in modifying your programming language from the beginning.  You decide this isn't worth your time, quit your job, and join _Quant2Bet_ instead who bought you out for $1,000,000.

# A Solution
[One VM to Rule Them All][paper] presents an architecture which allows implementing new programming languages within a common framework, allowing for language-agnostic optimizations.  Instead of designing the entire stack for a new programming language, you can just focus in creating a parser and AST interpreter.  Now, you can reuse functionality from existing Virtual Machines and your language is already fast and optimized.

## Background
* A __Java Virtual Machine (JVM)__ is a virtual machine that allows computers to run programming languages that compile to Java bytecode.
* An __Abstract Syntax Tree (AST)__ is a tree representation of a source code.  Each node is a specialized construct of the programming language, with branches as inputs to the construct.
* A __just-in-time (JIT) compilation__ compiles a program at run-time opposed to prior execution. JIT compilers rely on Ahead-Of-Time (AOT) compilation and interpretation.

## One VM to Rule Them All 
This paper claims the following:
* A method of rewriting nodes where a node can rewrite itself to a more specialized or general node
* An optimizing compiler that takes in the structure of the interpreter
* Speculative Assumptions and deoptimization to produce efficient machine code.  In particular, when speculative assumption fails, deoptimization is used to continue execution using interpretation 

The combination of these claims result in high-performance from an interpreter without the need of implementing a language-specific compiler.

In their prototype, they call the language implementation framework _Truffle_ and the compilation infrastructure _Graal_, which are both [open source][graal] by Oracle.  At a high level, a user of this ecosystem implements an AST interpreter for the guest language.  In the interpreter, each node encapsulates information regarding a particular semantic of the guest language.  The paper describes that the semantics for addition are described in an addition node.  Node rewriting occurs during interpretation, using profiling feedback.  When a subtree of the AST is deemed to be stable (meaning unlikely to be rewritten), the AST is partially evaluated at that subtree and the Graal compiler produces optimized machine code to run on the VM.  

If during execution a node is not the correct specialized type, we _deoptimize_.  The optimized machine code is discarded and we switch to AST interpretation.  The node rewrites itself and the subtree is then recompiled.  

Here, dynamic compilation is agnostic to the semantics of the guest language.  

The below diagram describes an instance of an AST interpreter during execution.  In Figure 1, nodes are first uninitialized.  After profiling feedback, nodes are rewritten to become specialized and are then compiled.

Now suppose in Figure 3, there is integer overflow.  Our speculation that all nodes in that subtree are of type integer is incorrect, so we have to deoptimize, rewrite a second time, and then recompiled.  Note that node rewriting is not just for type transitions, but for any time of profiling feedback.

< INSERT FIGURE 2 and 3>

## Node Rewriting

It is the duty of the developer of the guest language to implement node rewriting.  The paper urges the developer to fulfill the following conditions

> __Completeness__ - Each node must provide rewrites for _all_ cases that it does not handle itself.

> __Finiteness__ - The sequence of node replacements must eventually terminate to either a specialized node or a generic implementation that encompasses all possible cases.

> __Locality__ - Rewrites occur locally and only modify the subtree of the AST.

Examples of profiling feedback are type specializations, polymorphic inline caches and resolving operations.  As program is successively executed, the profiler yields more optimized compiled code.  The authors then claim that because of this, their interpreter is better than other interpreter implementations.  They note that the main overhead is dynamic dispatch.

Consider the following example, where the guest language is JavaScript:

```
function sum(n) {
	var sum = 0;
	for (var i = 1; i < n; i++) {
	sum += i;
	}
	return sum;
}
```
The following is an example of an AST after immediate parsing.  Note that only constants are typed.

< INSERT FIGURE 5>


In code, we can write the integer addition node as follows.  Here, Java is the host language for the JavaScript interpreter.

< INSERT FIGURE 6 >

After execution, nodes replace themselves as specialized nodes for type _integer_ called IAdd to be used for subsequent executions.  Note that IAdd nodes only operate on integer values.  If it does not receive an integer value, it will throw an exception, and the node will be rewritten.  

The below picture shows an instance where nodes are specialized for the integer type (depicted by prefix "I").

< FIGURE 7 >

If `sum` overflows, we need to rewrite certain nodes to specialize for the double data type.  The following image shows this case:

< FIGURE 8 >

We can actually do a bit better in code if the host language is Java.  For a given node, in this case `add`, we can use the annotation `@Specialization` to denote a specialized implementation and use `@Generic` to denote the default, unspecialized implementation.  Now the Java compiler will call the Java Annotation Processor which over all Node Specifications, marked by the annotations.  It is essentially the same as before though now we can use developer tools and IDEs.  

## Performance
The main overhead is dynamic dispatch between nodes when rewriting occurs.  To do this, we count the number of times a tree is called, and when it exceeds a certain threshold, the tree is assumed to be stable and is then compiled.  Deoptimization points invalidate the compiled code, allowing for the node to be rewritten.  The counter is reset, and after the threshold number of executions it will be deemed stable and compiled again. 

This architecture allows for additional optimizations, such as:
> __Injecting Static Information__ : where a node adds a guard that leads to a compiled code block or a deoptimizing point
> __Local Variables__ : where read and write operations on local variables are delegated to a Frame object that holds values.  [Escape Analysis][36] is enforced allowing static single assignment (SSA) form.
> __Branch Probabilities__: where probabilities of branch targets are incorporated to optimize code layout which decrease branch and instruction cache misses.  


# Current Implementation and Deployment
Currently, this infrastructure is prototyped in a subset of Java.  

< INSERT FIGURE 17 >

The __Truffle API__ is the main interface to implement the AST.  The __Truffle Optimizer__ involves partial evaluation.  The __VM Runtime Services__ provides basic VM services which includes Graal.  

The authors suggest two main deployment scenarios:
> __Java VM__:  The current implementation is in Java so it can technically run on any Java VM.  This is especially useful for debugging and low cost.
> __Graal VM__:  This provides API access to the compiler, so the guest language runs with partial evaluation.  This uses Graal as its dynamic compiler and is extremely fast.  It is useful for integrating the guest language in a current Java environment.

# Merits and Shortcomings:

This is an interesting idea that definitely is worth exploring when implementing the backend for a new programming language.  The Truffle and Kraal ecosystem are unlike others and provide 


# A Survey on Languages
The following languages have been tested by the authors:
> __JavaScript__:
> __Ruby__:
> __Python__:
> __J and R__:
>__Functional__:

# Related Work


























[paper]: https://dl.acm.org/citation.cfm?id=2509581
[graal]: https://github.com/oracle/graal
[36]: https://dl.acm.org/citation.cfm?id=1064996