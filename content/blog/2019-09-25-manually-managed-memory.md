+++
title = "Manually Managed Memory in Bril"
extra.author = "Drew Zagieboylo"
extra.author_link = "https://www.cs.cornell.edu/~dzag/"
extra.bio = """
  [Drew Zagieboylo](https://www.cs.cornell.edu/~dzag/) is a 3rd year PhD student researching Security, HW design and Programming Languages. He enjoys rock climbing and gaming in his free time.
"""
+++

- What was the goal?

Our goal was to add *pointer types* to Bril. _Pointers_ represent references to manually managed read/write memory cells which can persist outside of function scope. Furthermore we support C-style arrays such that pointer arithmetic instructions can be used to index into allocated memory regions. Lastly, we wished to ensure that value typechecking was still supportable for our new instructions (however we did not implement a typechecker). Our pointer types are meant only for value checking (i.e. every pointer type totally specifies the type of its contents); they do not include bounds or alias information to prevent memory safety bugs.

- What did you do? (Include both the design and the implementation.)

First we expanded the Bril syntax to support pointer types of the form ```ptr<TYPE>```. Pointers may represent references to data of any type, including other pointers. Therefore the type ```ptr<ptr<bool>>``` is well-formed, while the type ```ptr<ptr<>>``` is ill-formed.

Next, we modified the interpreter to include instructions for pointer allocation and deallocation of the form `dest: ptr<TYPE> = alloc arg1` and `free arg1`.

- What were the hardest parts to get right?

- Were you successful? (Report rigorously on your empirical evaluation.)