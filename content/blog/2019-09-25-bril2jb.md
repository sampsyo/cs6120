+++
title = "bril2jb: A tool that translates bril to Java Bytecode"
extra.author = "Siqiu Yao"
extra.bio = """
  [Siqiu Yao](http://www.cs.cornell.edu/~yaosiqiu/) is a 3rd year Ph.D. student interested in security about systems and programming languages.~~~~~~.
"""
+++

### Goal
The goal of this project is to provide a tool to translate bril code to Java Bytecode.

### Design
#### Before the actual translation process
At first glance, translating bril to [Java Bytecode](https://en.wikipedia.org/wiki/Java_bytecode) seems a 
pretty straightforward job, since bril (currently) only contains simple instructions and supports no function invocation.
But it turns out that making something directly runnable by *Java Virtual Machine*(JVM, the run-time environment of Java Bytecode)
is not trivial, because we need to construct our program as a Java *class*, 
which is the only code format JVM accepts.

Therefore we need to create a wrapper class(say `Wrapper.class`) like the following :
```JAVA
public class Wrapper {
  public Wrapper();
  public static void main(java.lang.String[]);
}
```
And the `main` function would contain the instructions in the bril code.

#### The translation process
I chose [ASM](https://asm.ow2.io), a Java Bytecode manipulation and analysis framework to help with the translation.

The translation process is pretty regular. One interesting point is that Java Bytecode is strictly-typed, 
while bril is naturally dynamic and 
there might be no explicit declaration of a variable before it is used(in the order of instruction list),
and some instructions are polymorphic such as `print`.
Therefore, we preprocess the instructions first to gather all type information and labels.
This step also infers types of variables, so there is actually no need to indicate types in bril anymore.
  

### Implementation
This tool is implemented in Java, and therefore maintained in a standalone repo. It translates bril code(in `json` format) to 
a same-name Java class file, which can run on JVM. 
For the usage guidance of our tool, please read `README.md` in [our repo](https://github.com/Neroysq/bril2jb).

We assume that the input program is valid and has no faulty behaviors 
because we think error-handling is not the topic of this project.   

There are some tricky details:

1. Since `int` in bril is 64-bit, all `int` variables are `long` variables in JVM; 
there is no boolean type in JVM, so all `bool` variables are `int`(32-bit integer) type in JVM.

2. ASM handles the max local variable size,
 but the index of local variable needs to be manually maintained(`long` type takes 2 index units while `int` takes 1). 
 
3. The hardest part of this project, in my opinion, is translating `print`. 
The `print` instruction is translated as `java.lang.System.println`, 
and all arguments are converted to `String` then concatenated. 
So before invoking the print function, we need to invoke the built-in concatenation function (`java.lang.invoke.StringConcatFactory.makeConcatWithConstants​`) first,
this process involves dynamic method invocation in Java, 
which requires a static Bootstrap Method to create it dynamically and 
therefore supports dynamic descriptors and arguments formatter 
([further reading](https://www.guardsquare.com/en/blog/string-concatenation-java-9-untangling-invokedynamic)).
We need to generate a descriptor and formatter for each usage of `print`.

### Evaluation
We assume all programs are valid bril programs.

We tested our tool using manually-written test cases(including the ones in bril repo) to ensure the correctness.
All of our tests pass and randomly chosen output Java Bytecode to look reasonable.

### Conclusion
In conclusion, we successfully built a translator from bril to Java bytecode. 
I look forward to further maintaining this tool to support potential new features 
such as function call and memory allocation.
