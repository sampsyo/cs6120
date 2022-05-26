+++
title = "Xi++: Object Oriented Features for Xi"
[[extra.authors]]
name = "Michael Maitland"
link = "https://michaelmaitland.com"
[[extra.authors]]
name = "Orko Sinha"
link = ""
+++
Building a compiler for Xi is the main project in [CS 4120](https://www.cs.cornell.edu/courses/cs4120/) at Cornell. The final assignment, now discontinued, is to extend the compiler to support features of the [Xi++ Language Specification](https://www.cs.cornell.edu/courses/cs4120/2019sp/project/oolang.pdf) such as object-oriented programming. Our project is an implementation of these features on Michael's compiler created for CS 4120. Our project is [here](https://github.com/Yasgur99/xic/tree/oo). The changes introduced by this project work in the `IRSimulator` that was part of the code released by CS 4120, but also work for the x86 code generated.

# Lexing and Parsing
For lexing, there are 5 new syntax elements that we need to handle, listed below:
1. `class`
2. `extends`
3. `this`
4. `new`
5. `.` operator

These correspond to 5 new token types corresponding to each keyword in the language. The existing JFlex file created for the original project is extended to support these features.

A sample program that uses all of these new keywords is as follows:

```
use io

class B extends A {

  x : int

  init() : B {
    this.x = 2 
    return this
  }
}

f() {
  print(new B.init().x)
}
```

The parser handles the new semantics and allow parsing of the following:

1. Interfaces can now contain zero or more `use` statements. Use statements bring functions and classes in other modules into scope. In the example above, the `print` function is brought into scope from the `io` module.
2. Interfaces can now contain zero or more class declarations.
3. A class declaration consists of the class name, the class name that it extends if any, and zero or more function declarations. Note that fields are not defined in class declarations. We will discuss the impact of fields not being defined in the class declarations below.
4. Programs can contain zero or more class definitions.
5. Class definitions consist of zero or more fields and zero or more function definitions.
6. Allocation of a new object using the `new` keyword.
7. Ability to refer to the object `this` within methods.
8. Ability to make method calls which take the form `o.f()` or `this.f()`.
5. Ability to make a field access which takes the form of `o.x` or `this.x`

# Symbol Tables

We extend the old symbol table that previously included entries for function declarations into a symbol table that includes both function declarations, class declarations and object fields. Recall that fields are not defined in a class declaration. Instead, they are defined only by the class definition. This is why we must track object fields separate from class declarations. 

While the symbol tables are built up, we also check for a certain class of semantic errors. These types of errors include redefinition of the same class and redefinition of fields. Consider the following examples:

```
// A.ixi
class A {
  f();
  f();
  f(x : int);
}
```
The snippet above is an interface that erroneously declares `f()` twice and also erroneously defines an overloaded version of `f` that takes an integer as an argument. Currently, function overloading is not permitted by the semantics. 

```
// A.xi
class B {
  x : int 
  x : int 
  x : bool
}
```
The snippet above is a class definition that erroneously defines the field `x` twice with the same type and also erroneously defines x with another type.

When building these symbol tables, the code collects all errors and does the best it can to finish building the table. If there are any errors at the end of the entire semantic analysis, which includes type checking, all errors are reported and the compilation process is terminated. This allows the programmer to debug their programs much more efficiently. 

As a side note `.ixi` is the file extension for interfaces and `.xi` is the file extension for programs.

# Dispatch Vectors

Dispatch Vectors were the biggest focus during this project. According to Professor Myers, there are [3 goals](https://www.cs.cornell.edu/courses/cs4120/2022sp/notes.html?id=objects) we had:

1. Method calls should be fast, since they are the core operation in the language
2. The space overhead per object should be low. We are less concerned with space overhead on a per-class basis, since the number of objects is typically much larger than the number of classes.
3. We would like to share inherited code across classes, since for a large class hierarchy, we might otherwise end up with many copies of the essentially the same code, leading to the dreaded “code bloat”.

A straightforward approach that has been used in dynamically typed OO languages like Smalltalk is to simply search for the right method definition. Each object has a hidden field that points to its class object; a method call `o.m()` is implemented by walking up the class hierarchy, starting from `o`'s class, and stopping once an implementation for method `m` is found. This implementation allows sharing inherited code, so it avoids code bloat, but it is much slower than a simple function call.

For faster method dispatch, we need to precompute a data structure---a dispatch table. The key idea is to map different method names onto small integer indices, and to use those indices to look up the appropriate method code pointer in an array of code pointers. These arrays of code pointers go by various names. In C++, this array is called a [vtbl](https://en.wikipedia.org/wiki/Virtual_method_table), pronounced like “v-table”, standing for “virtual table". In other literature it is known as a selector table or dispatch vector.

So consider the following classes:

```
class A           { f(); g(); }
class B extends A { f(); h(); }
```

`A` and `B` should share the same call to `f`, `B` should inherit `g` from `A`, and `B` should also define `h`.

We can put `f` and `g` in indices 0 and 1 respectively for both dispatch vectors and `h` is put in index 2 for the dispatch vector of `B`. Then, if there is a call to `f`, `g`, or `h`, we can get the pointer to the correct function in `O(1)` time by generating IR that resembles the following:

```
MOV dv_ptr, o // Get the pointer to the dispatch vector since it is at index 0 of the object o
MOV dv, [dv_ptr] // Get the dispatch vector
CALL [dv + index] // Call a function based on its index in the dispatch vector
```

We build two classes to assist with building dispatch vectors:

First we need to figure out when given some class `A`, and some function `f`, which class's `f` should we call. For example, if `f` is redefined by a subclass then we want to call the `f` of that subclass. But if `f` is defined by a super class we want to call `f` of that super class. This is all handled by the `DispatchVectorClassResolver`.

The next thing we need is given all methods for a given class, what index do they go in in the dispatch vector. For example if `A` defines `f` and subclass `B` also defines `f` they must be at the same index. The `DispatchVectorIndexResolver` handles this.

The `DispatchVectorClassResolver` and `DispatchVectorIndexResolver` both take the symbol tables that were constructed during semantic analysis and calculate which class name owns the function that should be invoked and which index it is in the dispatch vector respectively. Then, the `NodeToIRConverter` can make calls to the revolvers to get this information on a method by method occurrence.

Both of these classes have been unit tested to see what happens in the following instances:
* No methods and no super class
* No method in the super class but the class defines a method
* No method in the class but methods in the super class
* Mix of methods and no methods in a class hierarchy of length 3
* Methods with the same name as super
* Methods with different names as super
* methods where some of the names are the same as super

When generating the actual IR for the dispatch vectors, we tried a few different approaches. First, we converted the mangled function names into strings by representing them as an array of characters in the IR, and then tried to call the function by loading the string. We quickly realized that this would be difficult because the strings are often multiple characters and when we do `call [dv + index]`, we're really calling a function with the name corresponding the the first character in the mangled function name. We quickly scrapped this approach and tried to build them using static data.

In Xi, static data is represented by an `IRData` class. Out of the box, `IRData` is not a traditional `IRNode` (it does not extend `IRNode`). This means its children nodes are not `IRNode`s. Instead, it has a field `long[] data`. We tried to encode the dispatch vector as the concatenation of function names separated by the null terminator. Again, we ran into the problem that when we do `call classNameDVLabel` it only reads the first word of data which corresponds to the first character. Even worse, the interpreter tried to jump to the instruction at the address corresponding to that character instead of looking up that character in the function definition to address table and jumping there. I wrote a version of the interpreter that read until the next null terminator and then did the correct lookup. This got things to worked but it felt hacky because we could no longer write IR that jumped to an address---it required we jumped to a function that was named as the string starting at that memory address until the next null terminator byte in memory.
 
Both of these approaches stem from a core conceptual confusion: we were putting strings where we wanted to put addresses.

We explored trying to put the memory address of the function in the `IRData` instead of the name. That way, we would only need to read one word and the interpreter would jump there as it wanted to do. The problem with this approach is that when we convert to the IR, we have no idea about the memory layout. The memory addresses are defined by the interpreter and would be different than the memory scheme if we went all the way to assembly. Additionally, there was no way to know about the memory layout of the simulator when converting to IR because it does not initialize its memory until after the IR is created.

We had two options: 

1. Modify `IRData` to allow us to have `IRName` or `IRConst` or `IRMem` nodes at each element in the data array instead of `long`s. 
2. Initialize a non-static dispatch vector for each object on construction so we could use `IRName`, `IRConst` and `IRMem` to represent the dispatch vector and offsets into it.

The first approach requires changes to the IR and the IR simulator. The second approach requires no changes to the IR or the simulator. We opted for approach 2 for simplicity, although approach 1 is better from an efficiently perspective. Approach 2 requires a new dispatch vector to be constructed every time we allocate a new object instead of reusing one copy for a given class type. If the dispatch vectors are large, this can become costly. We discuss this more in future work section.

# Desugaring
The Xi++ specification states that fields and methods in a class are brought into scope automatically. To supplement this, we refer to fields as `this.x` and methods `this.f()`.  However, as per the language specification, we also need to support `x` and `f()` respectively from the examples prior. This is syntactic sugar. The `ThisAugmentor` created is a visitor that converts unqualified field accesses and method calls to qualified ones. Simply, it takes `x` or `f()` into `this.x` or `this.f()` respectively.

## `ThisAugmentor`
This class keeps track of the current class (since we do not allow nested classes yet, but when we do we will just keep track of stack of classes), and a stack of contexts. We enter a new context when we see a new block of code, a new function, or enter into a new loop block. Any declarations or function arguments add a new binding to the current context. Then the `ThisAugmenter` can check to see if the binding is in a context. If it is, we know its not a field because the variables in the context would shadow all the fields. But if its not in any of those contexts, then we can lookup the fields in the current class context.

### Function Expressions

A function expression corresponds to a function call. For example, `x = f() + 1` is an assignment that contains a function expression `f()`. If a function expression `f()` does not occur within a function definition within a class (a method), then we know that no augmentation is needed since `this` can only be used in methods. Otherwise `f()` is occurring in a method. If there is a function with the name `f` in the class it is being called in, then we must convert it to a method call where the object of the method call is `this`.

### Function Declarations, Function Definitions, Function Expressions

Methods take a reference to the object as a hidden first argument. If the function declaration is a method we add this as the first argument. A function definition consists of a function declaration and a function body in a program. A function declaration can also exist without a body in an interface. Therefore, as long as we augment all programs and all interfaces, all function declarations will agree.

Consider the following example:

```
// A.ixi
class A {
  f()
  g(x:int)
}
// A.xi
class A {
 f() { ... }
 g(x:int) { ... }
```
will be augmented to 
```
// A.ixi
class A {
  f(this : A)
  g(this : A, x:int)
}
// A.xi
class A {
 f(this : A) { ... }
 g(this : A, x:int) { ... }
```

Notice that since type checking has already occurred before augmentation, then after we augment we do not introduce any new semantic errors since `f` and `g` got the augmented in both the interface and the program.

For all function expressions that are method calls, qualified or unqualified must add this as the first argument. 

Consider the following example:

```
class A {

 f() { ... }

 g() {
   f()
 }

 init() { return this }
}

h() {
 a : A = new A.init();
 a.f()
}
```
will be augmented to 
```
class A {

 f(this : A) { ... }

 g() {
   f(this)
 }
}

h() {
 a : A = new A.init();
 a.f(a)
}
```

### Variables

We must lookup to see if the variable is a field and if it is we wrap it in a field access. For example if `x` is a field, then it must become `this.f`.

### Declarations

When we see a declaration we must add a new binding in the variable context. This is needed in case there is anything that should shadow the scope of a field. Consider the following example:

```
class A {
  x : int

  f(x : int) : x {
    return x     
  }
}
```

In `f`, `x` is a variable local to that function call and `x` is also a field. In this case, we should not augment to `this.x` since the intended behavior is to refer to the `x` as defined in the function argument.

Our project also supports declaration shadowing where the declarations have different types than the field that it shadows.

# IR Generation
We introduce three new changes to the IR:

1. Addition of `IRClassDefn` which represents a class name and the list `IRFuncDefn` which represents its methods.
2. Add list of `IRClassDefn` to `IRCompUnit`. The `IRCompUnit` represents the root of the IR tree. It also contains all the functions that are not methods, static data, and functions to be executed by the runtime before calling the main function, similar to [crt0](https://en.wikipedia.org/wiki/Crt0) in the C language.

## Function Definitions and Class Definitions

Function definitions must be converted differently depending on whether that function is a method or not. When we convert a function that is a method to IR we must treat it differently than a normal function because the ABI naming convention is different for functions and methods. We will discuss this convention below. But because of the difference between functions and methods, we do not convert to IR for methods and let the owning class definition do the conversion. 

## Field Access

An object is represented as an array of word sized entries. The first entry is a pointer to the dispatch vector. The subsequent entries are fields in the order that they were defined in the class definition. Therefore, we can convert a field access by treating it as indexing into an array at the correct offset.

## Method Call

We lookup the function in the dispatch vector and generate the same IR as a normal function expression except that a function expression  uses a name but we are using a memory address that contains the name. We don't need to worry about passing the object to the function because the `ThisAugmenter` already took care of this for us.

## New

We must allocate the new object and then call the initialization function. First we build the dispatch vector and get a pointer to it. Then we convert all the fields to IR. Then we create an array that contains this pointer followed by the fields. Lastly, we call the initialization function, passing the reference to the new object as the first argument.

## `NodeToIRConverter`

We introduce a few new fields for bookkeeping:

1. `objectMethodEncodings` to store mangled function names. This allows us to store a mapping from method name to the actual ABI encoding for that function.T he encoding is the same as the encoding defined for a function or procedure in the Xi ABI except that the function name is proceeded by the class name and an underscore. See the section on `DispatchVectorClassResolver` for more information.
2. `classNameToNumFields` represents a mapping from class name to the number of fields that class has.
3. `classNameToFieldNameToIndex` represents a mapping from class name to the index of the field in the object representation.

This class is also responsible for building the dispatch vector and getting the method symbol out of a dispatch vector.

# Future Work

## Type checking

This project skipped over the type checking phase so that more time could be spent focusing on the actual object oriented features. Obviously, this allows Xi programs that are not well typed and would likely crash or misbehave if these types of programs are used. Additionally, type checking would allow programs that involve more than one class. At the moment, it is impossible to figure out what class a variable `o` belongs to in situations such as `o.f()` and `o.x`. Currently, they are hard coded to assume that all objects are of type `A`. Type checking would allow the compiler to identify what type a variable has and therefore make decisions based on class layout for any class type.

## Fields

All fields are private to a class. This means that if a class `A` has field `x` and class `B` extends `A`, then class `B` cannot refer to `x`. However, `B` can call methods of `A` and those methods may refer to `x`. Currently, subclasses do not allocate fields that belong to their parent classes. Therefore when a method of a super class is called, it has no way to get those fields. In the current implementation it will try and get the variable from the expected index which will really be an undefined memory access.

In order to remedy this problem, we must allocate fields belonging to super classes at runtime since fields are not part of class declarations and are only part of class definitions by the Xi++ specification. This means that we could have pre-compiled code for module `A` and code that imports `A` with a `use` statement. The problem is that the interface for `A` does not mention any fields that `A` has. Therefore, if our code creates a `class B extends A`, the compiler has no idea how to create an object B that has the fields that belong to `A`. C++ solves this problem by requiring public and private fields to be declared in the header files (the class declaration) so this information is known at compile time. Java on the other hand does this at runtime by having a byte code instruction that refers to the variables index. In Xi, we could accomplish this by creating a static table that allows us to determine what index a field name is at, similar to Java's approach.

## Static Dispatch Vectors

As discussed above, much of the work that went into this project was with respect to the representation of dispatch vectors in the IR. I met with Professor Myers and he agreed that his intention is for these dispatch vectors to be static data but there is no way to represent them using the current version of `IRData` since `IRData` requires the physical address of functions but these are not determined until runtime. The solution is to create a different `IRData` class which can represent static data that is of the form `IRNode` instead of physical addresses. This would allow us to say the data is an array of `IRName`s which would be converted to the address of the correct function at runtime. Compared with the solution we chose to implement, this would create a single dispatch vector for each class definitions, while our solution creates a dispatch vector for every single object allocated. The time it takes to create one dispatch vector as static data is proportional to the number of class definitions while our solution must take time to create it for every single object allocated. In terms of implementation effort, the actual conversion to IR is equivalent, but modifying the `IRData` class, modifying the interpreter, and testing the interpreter for those changes was the real effort.

## Private Methods

The specification of Xi++ allows methods to be defined in class definitions that are not part of class declarations. The current implementation expects that this would result in a type checking error since the declaration and definition do not match. However, we could imagine that a class definition declares methods that it does not want to export. These types of functions could be represented in the object but not part of the dispatch vector. An alternative approach would be to store it in the dispatch vector but type checking would not permit `MethodCall`s outside the class to use it nor allow subclasses to use it.

## Evaluation
In addition to unit tests for Lexing, Parsing, `NodeToIRConverter`, `ThisAugmenter` and dispatch vector tests we also have a series of integration tests that combine these features into the Xi/Xi++ compiler.

These integration tests include:
1. `class_equality.xi` - Checks if an object is equal to itself and if two objects are not equivalent to each other by using the `==` operator 
2. `class_field_access.xi` - Checks a series of class field accesses with inheritance
3. `class_method.xi` - A test for methods working for classes using a field
4. `field_access.xi` - A test for fields working for classes
5. `method_call.xi` - A test for methods working for classes without a field
6. `point.xi` - A comprehensive test using inheritance and various field and methods of classes
