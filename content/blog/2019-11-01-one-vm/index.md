+++
title = "An Efficient Implementation of Self"

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
* Speculative Assumptions and deoptimization to produce efficient machine code.  In particular, when speculative assumption fails, deoptimization is used to continue execution using interpretation.

The combination of these claims result in high-performance from an interpreter without the need of implementing a language-specific compiler.

In their prototype, they call the language implementation framework _Truffle_ and the comiplation infrastructure _Graal_, which are both [open source][graal] by Oracle.  We will first explore this idea at a high level, and then delve deep into their implementation. 






[paper]: https://dl.acm.org/citation.cfm?id=2509581
[graal]: https://github.com/oracle/graal