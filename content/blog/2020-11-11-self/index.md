+++
title = "An Efficient Implementation of SELF, a Dynamically-Typed Object-Oriented Language Based on Prototypes"
[extra]
bio = """
Evan Adler is a Masters of Engineering student at Cornell interested in compilers and programming languages. 
"""
[[extra.authors]]
name = "Evan Adler"
latex = true
+++

## Background

This week, we discuss the paper ["An Efficient Implementation of SELF, a Dynamically-Typed Object-Oriented Language Based on Prototypes"][efficientself] (Chambers, Ungar, and Lee), which appeared in OOPSLA 1989. As it turns out, the same year which heralded the downfall of communism in Europe also marked a significant victory in the liberation of object-oriented programming languages from class-based domination. What I mean is, as we'll see, Self is the first object-oriented programming language without classes, something not really considered possible before its debut.

Over the years Self has been a research endeavour at three of the most important institutions in the history of Computer Science: Xerox PARC, Stanford University, and Sun Microsystems. As far as I can tell, this week's paper is the fourth Self paper to be published. The first, [SELF: The Power of Simplicity][self] (Ungar and Smith), written by the language's creators, introduced Self in 1987. The second, [Object Storage and Inheritance for Self][leethesis] (Lee), was written in 1988 and discusses Self's object layout (at that point in time) and Self's multiple inheritance rules. The third, [Customization: Optimizing Compiler Technology for Self, a Dynamically-Typed Object-Oriented Programming Language][customization] (Chambers and Ungar), written earlier in 1989, covers some of the same optimizations discussed in the Chambers, Ungar, and Lee paper.

### Smalltalk

It needs to be noted that Self is based on Smalltalk, a language which influenced many of the modern programming languages we know and love. I recommend reading [the Wikipedia page on Smalltalk][smalltalkwiki] before the Self paper. Here I'll highlight the key features of Smalltalk which are needed to understand Self.

* Classes<br>
Every object is created from a class constructor. However, [classes themselves can be modified at runtime][sosmalltalk]. Also, under the hood, classes are just a special type of object (this leads naturally to the concept of prototypes that Self pioneered).

* Message Passing<br>
The way to interact with objects is to send them messages. At runtime, the "receiver" object scans itself and its parent classes for a method with the same name as the message, and if one exists, it executes it. The message can also contain method arguments. This can be referred to as [single dynamic dispatch][dispatch] or [duck typing][duck]. In fact, this is the way in which Smalltalk and Self are dynamically typed languages.

* Blocks<br>
Blocks of Smalltalk code (i.e. the things enclosed by curly braces in most languages) are themselves objects and can be passed as values. They can also be given parameters, which facilitates their use as lambda functions.

* JIT<br>
Just In Time Compilers [can trace a lot of their history][jit] through Smalltalk, Self, and later Java. All three of these languages compile into an intermediary bytecode and translate into machine code at runtime.

### Self

Self prides itself for its simplicity. I think this is because, while we could think of it as adding prototypes to Smalltalk, it can be equivilently thought of as removing classes from Smalltalk, which is a minimalistic language to begin with.

Self introduces prototypes, which is a way to have object oriented programming without classes. Instead of calling class constructors, objects are created by cloning other objects (called the object's prototype) and potentially modifying their structure. The predominant modern adopter of prototypes is [Javascript][js].

Self objects constist of named "slots". Slots are object pointers which can reference class fields, method objects, or parent objects (Self uses multiple inheritance).

If you have an object `o` with a slot named `foo` that contains 5, and you send `o` the message `foo`, you get 5. If `o` has a slot named `bar` that points to a method object `f(x)` that computes `foo*x`, and you send `o` the message `bar(3)`, you get 15. How can `f` refer to `foo` if the `f` object does not inherit from `o`? The answer is that when you send the message `bar(3)`, a new method object is created, using `f` as a prototype, which has a new parent slot pointing to `o`. The name of this new parent slot is ... "self".

In the above example, the message `foo` acts like a "getter." If you want to provide a "setter" for the `foo` slot, you can give `o` a special "assignment slot" which behaves like a setter method. One really cool thing about Self is that you're not limited to just providing these setters for data fields. You can also provide assignment slots to methods and parent objects so that they can change dynamically at run-time!

Back to our example with `o`, as I mentioned, when the message `bar(3)` is sent, `f`'s `self` slot is rebound dynamically. In contrast, there is a special object called a block which points to a method. The `self` field of that method does not get rebound dynamically, but is rather fixed. This corresponds to a lexically scoped method or a closure, which matches Smalltalk's blocks as you may recall.

Self also has object-array and byte-array objects.

## The Object Storage System

The authors quickly mention that they borrow the following two implementation details from the state-of-the-art Smalltalk implementations of the time:

* Avoiding an [object table][objecttable] (a piece of pointer indirection which speeds up garbage collection but slows down each object reference and requires space).
* Using a garbage collector based on "Generation Scavenging with demographic feedback-mediated tenuring, augmented with a traditional mark-and-sweep collector to reclaim tenured garbage". It sounds like these guys had enough garbage industry know-how to put Tony Soprano out of business!

### Maps

Now on to the novel ideas of this paper. The authors point out that without classes, similar objects duplicate a lot of data. To save space, they define clone families and maps. A clone family is a set of objects with the same structure. That is, if you create object `A` by cloning `B` and you only modify the values of `A`'s assignable slots, then `A` and `B` are in the same clone family. Now, objects are no longer implemented "under the hood" as named slots, but rather as a pointer to the map of its clone family, which contains all the shared structure, as well as its assignable (field, method, or parent) pointers. However, the pointers are not named. Instead, the ordering and naming information resides in the map. Maps make the memory footprint of Self programs resemble that of class-based Smalltalk programs.

### Segregation

This was a pretty interesting compiler optimization. Often, the heap is traversed and all references need to be processed. This is something the garbage collector does, something that needs to happen when an object is moved due to its size increasing, and something the "browser" does, which I believe is part of the Self IDE. Now, traversing through the heap looking for pointers becomes problematic when part of a byte-array happens to look like a memory address. Since byte-arrays are the only construct with this ambiguity in both Smalltalk and Self, Smalltalk implementations would traditionally have some extra overhead to check for byte-arrays as it processes the heap, so it knows to ignore pointer-like words which are part of a byte array.

Instead, Chambers, Ungar, and Lee segregate the heap into sections where byte array objects can be allocated and sections where all other objects can be allocated. This way they avoid the overhead of being "on the watch" for byte arrays as they scan the heap.

They also use two other tricks to speed up heap scans. First, at the end of the heap region, they place a dummy "sentinel" reference that gets the attention of the scanner. This way, the scanner only needs to check if it's within bounds when it sees a reference instead of every time it advances itself by one word. Similarly, a scanner often needs to capture the object which references another object. To minimize bookkeeping, every object begins with a special tag called a "mark". Then, when a reference is found, the scanner can just iterate backward to the closest mark to find the referencing object.

Segregation, sentinels, and marks improve from the heap scanning speed of the fastest Smalltalk implementation by a factor of two.

### Object Formats

32-bit words in the slot contents can be one of three things: 30-bit integers, 30-bit floats, or 32-bit addresses. The two least significant bits encode which type of data the word represents. As far as the memory address is concerned, this type of encoding is an example of a [tagged pointer][taggedptr]. The last two bits are also used to differentiate the "mark" words described in the previous section, although these words are never placed inside slots.

This section of the paper goes on to show how objects and maps are layed out in memory. I don't think there's anything too interesting or clever about this, but the kicker is that at the end the authors can demonstrate concretely that by using maps, their Self implementation is as space efficient as Smalltalk.

## The Parser

Self uses a JIT, and therefore Self code is compiled into bytecode. Method objects store the bytecode as data using a byte array and an object array. Self uses a very simple stack-based bytecode representation with only eight opcodes (four of them being used to originate or delegate messages). In contrast, Smalltalk-80 has a much more eleborate bytecode representation.

## The Compiler

Like the state-of-the-art Smalltalk implementation of the day, the Self implementation uses *dynamic translation* (again, JIT), and *inline caching* of message results.

Smalltalk programs are notoriously hard to optimize because the types of everything except primitives are not really known statically. Self has the same issue, compounded by the fact that variables are only accessible through message passing, which further obfuscates the compiler's view. In addition, Self cannot use Smalltalk optimizations which operate on classes. Nonetheless, the novel techniques described in this section make Self twice as efficient as the state-of-the-art Smalltalk implementation, or about five times slower than optimized C. I will point out that later implementations of Self acheived speeds only twice as slow as optimized C!

### Customized Compilation

To illustrate this optimization, the authors have us consider a `min` method which calls a `less-than` method. Both the `min` and `less-than` methods are potentially called from many types of objects. A traditional Smalltalk JIT compiler would translate `min` into machine code when it's called on an integer, and reuse the compiled method when called on a floating point. This means `min` needs to be compiled in a type-agnostic way, which prevents Message Inlining (which we'll see shortly) of `less-than`.

However, with customized compilation, an integer-specific version of `min` is compiled when it's called on an integer, and a float-specific version is compiled when it's called on a float. This allows the `less-than` message-passing to be inlined...

### Message Inlining

Remember, the idea of message passing is that the receiver searches itself and its parents for the required slot name, and does something depending on the type of the slot (data, assignment slot, method, block). However, if the type of the receiver is known at compile time (remember what "compile time" means for a JIT), and the message is also known at compile time, the compiler can search for the relevant slot itself so that this dynamic dispatch doesn't have to occur at run-time.

### Primitive Inlining

This optimization inlines calls to builtin primitive operations such as arithmetic or comparison. It also constant-folds the calls if the arguments are known at compile time. In many cases, Customized Compilation enables Primitive Inlining because the types of the arguments are known statically.

### Message Splitting

This is an optimization that occurs at merge points in the control flow graph of a method being compiled. Say that basic blocks A and B both connect to C, and that C passes a message to some object `o`. `o`'s type may depend on whether the A or B path was taken. The optimization is to pass the message to `o` at the end of A and B instead of in C. This makes `o`'s type knowable at both points and enables Message Inlining.

### Type Prediction

This compiler optimization builds upon Message Splitting. The motivation here seems to be mostly based on Self's operator overloading. The idea is that a message like `less-than` will often be passed to integers, floats, and strings because the language provides this behavior, but in rare cases, this message could be passed to arbitrary objects. Then, any time the compiler sees a `less-than` message passed to an object `o`, it inserts branches that check if `o` is an integer, a float, a string, or something else. Then it Message Splits the single message-pass to the ends of these four branches. This allows Message Inlining in the common case and actual message passing in the worst case.

## Supporting the Programming Environment

### Support for Incremental Recompilation

This section of the paper addresses how the Self implementation uses dependency links between object maps and methods so that the minimal correctness-preserving adjustment can be made when either a method or object is changed by the programmer. This is like a really fine-grained version of a typical C/C++ Makefile, where dependency links are typically tracked for entire compilation units (files).

One interesting point here is that Message Inlining creates delicate dependencies between object formats and method code. This is because if an object map changes, the compile-time search for the correct slot may be invalidated.

### Support for Source-Level Debugging

This section explains how extra state is maintained to construct the "virtual callstack" (the one the programmer probably has in mind) from the "physical callstack" (which excludes the inlined methods), and also how to resolve at which program counter to insert a breakpoint.

## Performance Comparison

There's a lot to unpack here, and to save some fuel for discussion, I'll just outline a few important points.

* The authors acheive roughly a 2x speedup from the fastest Smalltalk-80 system, both on large standard C benchmarks (which are transliterated into Self and Smalltalk), and on a handful of small programs which are manually written in Self.
* The authors propose a new metric for measuring performance of Smalltalk-like languages, *Mims* (millions of messages per second) and a notion of *efficiency* (messages per instruction).
* Self is also compared to two versions of Smalltalk which include type declarations, and it acheives roughly the same performance.

## Merits and Shortcomings of the Paper

I won't do anything other than heap praise on the Self implementation described in the paper. There are a lot of great ideas, some of which advance prototype-based languages and some of which improve upon Smalltalk-like languages.

One nitpicky thing is, and I'm not sure what kinds of benchmarks were available in 1989, but the authors write that the Stanford integer benchmarks are 900 lines of Self code and the Richards benchmark is 400 lines. This seems pretty skimpy compared to something like the SPEC benchmarks.

## Questions

* In the Conclusions section, the authors emphasize that they acheived good performance without resorting to evil type declarations. However, Java, arguably a spiritual successor to Smalltalk and Self, opts for type declarations, presumably to squeeze out more performance. What are the flexibility vs performance tradeoffs inherent with type declarations?
* Which is superior, classes or prototypes? Why do we see so few languages implement prototypes?
* Self is designed with simplicity in mind. This can be seen in the "everything is an object" policy and the minimalistic bytecode. What are the advantages of a simple language? Does it boil down to the [Principle of Least Astonishment][astonishment]?
* This paper doesn't elaborate on Self's dynamic inheritance (changing a parent-object at runtime). What do you think of this feature?
* What are some shortcomings of the paper? (I'm going to punt this one.)

[efficientself]: https://dl.acm.org/doi/10.1145/74878.74884
[self]: https://bibliography.selflanguage.org/_static/self-power.pdf
[leethesis]: https://bibliography.selflanguage.org/elgin-thesis.html
[customization]: https://dl.acm.org/doi/10.1145/73141.74831
[smalltalkwiki]: https://en.wikipedia.org/wiki/Smalltalk
[sosmalltalk]: https://stackoverflow.com/questions/4460991/how-can-i-add-methods-to-a-class-at-runtime-in-smalltalk
[dispatch]: https://en.wikipedia.org/wiki/Dynamic_dispatch#Single_and_multiple_dispatch
[duck]: https://en.wikipedia.org/wiki/Duck_typing
[jit]: https://en.wikipedia.org/wiki/Just-in-time_compilation#History
[js]: https://medium.com/madhash/understanding-prototypes-in-javascript-e466244da086
[objecttable]: http://www.mirandabanda.org/bluebook/bluebook_chapter30.html#TheObjectTable30
[taggedptr]: https://en.wikipedia.org/wiki/Tagged_pointer
[astonishment]: https://news.ycombinator.com/item?id=22794771
