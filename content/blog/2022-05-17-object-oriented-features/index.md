+++
title = "Xi++: Object Oriented Features for Xi"
[[extra.authors]]
name = "Orko Sinha"
link = ""
[[extra.authors]]
name = "Michael Maitland"
link = "https://michaelmaitland.com"
+++
Building a compiler for Xi is the main project in [CS 4120](https://www.cs.cornell.edu/courses/cs4120/) at Cornell. The final assignment, now discontinued, is to extend the compiler to support feaures of the [Xi++ Language Specification](https://www.cs.cornell.edu/courses/cs4120/2019sp/project/oolang.pdf) such as object-oriented programing. Our project is an implementation of these features on Michael's compiler created for CS 4120. Our project is [here](https://github.com/Yasgur99/xic/tree/oo).

# Lexing
For lexing, there are 5 new syntax elements that we need to handle, listed below:
1. `class`
2. `extends`
3. `this`
4. `new`
5. `.` operator

These correspond to 5 new token types corresponding to each keyword in the language. The existing JFlex file created for the original project is extended to support these features.

# Parsing
The parser handles the new semantics and allow parsing of the following:
1. Parsing of `Interface` that contain zero or more `use` statements and zero or more class declarations. A class declaration is the name of the class, optionally the class name it extends, and zero or more function declarations. `ClassDecl` is created as a new AST node to represent this.
2. Parsing of `Program` that contain class definitions and function definitions. Programs can now contain one or more functions definitions and/or class definitions. A class definition is the name of the class, and the class it extends a list of fields which are `SimpleDecl`s, and a list of function defintions. However, the list of fields or methods can be empty. `ClassDefn` also represents this.
3. Parsing of `Exp` with the `new` and `this` keyword.
4. Parsing of `MethodCall`s which take the form `o.f()` or `this.f()`
5. Parsing of `FieldAccess` which take the form of `o.x` or `this.x`

Additionally, two AST nodes are added to handle Interfaces and Classes.
1. `InterfaceBody`
2. `ClassBody`

Previously, an `Interface` and `Program` contained the function declarations and definitions respectively. Previously, the parser would add to the lists directly when it came across a declaration or definition. Now that we are also handling class declarations and definitions, the parser has `fdecl_or_cdecl_star` and `fdefn_or_cdefn_plus` nonterminal to account for this. These two new nonterminals correspond to the two new AST nodes. This does not have a big input on the rest of the compiler as we just need to add a `getBody()` call in certain places that expected the function declarations and definitions to be accessed directly from the `Program`.

We also add AST nodes `FieldAccess`, `MethodCall`, and `New`.

# Symbol Tables

We extend the old symbol table that just included entries for function declarations into a symbol table that includes both function declarations, class declarations and object fields

## Implementation

The old symbol table was represented as a `Map<String, FunctionDecl>`. Now, we introduce a class `SymbolTable` that really serves as a wrapper. It has has fields `Map<String, FunctionDecl>` and `Map<String, ClassDecl>`. By using a wrapper, we can pass around the `SymbolTable` as a single object and we shift some of the operations on the map into our new abstraction.

This change required us to change the `SymbolTableManager` to operate on `SymbolTable` and do logic related to merging class symbols.

It also requires a renaming of `FunctionCollector` to `SymbolCollector` which extends the logic of collection function to also collect classes. We use the same visitor pattern where all `ClassDecl` nodes tell the visitor to extract their declaration out of them. All other nodes that are not `ClassDecl`s do nothing when the visitor visits them.

The `getValidPrograms()` function now returns a `ValidProgram` and takes one less argument. Previously it took an argument that the function mutated since it already had a different return value. I decided to create the `ValidProgram` class which lets us store both the old return value and this list of function declarations in one object. However after extending with class symbols we need a list of `FunctionDecl` and `ClassDecl`. A new object to represent this is `ProgramSymbols` which is really a wrapper over both lists.

# Desugaring
The Xi++ specification states that fields and methods in a class are brought into scope automatically. To supplement this, we refer to fields as `this.x` and methods `this.f()`.  However, as per the language specifiction, we also need to support `x` and `f()` respectively from the examples prior. This is syntactic sugar. The `ThisAugmentor` created is a visitor that converts unqualified field accesses and method calls to qualified ones. Simply, it takes `x` or `f()` into `this.x` or `this.f()` respectively.

## `ThisAugmentor`
This class keeps track of the current class (since we do not allow nested classes yet, but when we do we will just keep track of stack of classes), and a stack of contexts. When we enter a new context which occurs when we see a new block of code, new function, or enter a new loop block. Any declarations or function arguments add a new binding to the current context. Then isField function can check to see if the binding is in a context. If it is, we know its not a field because the variables in the context would shadow all the fields. But if its not in any of those contexts, then we can lookup the fields in the current class context.

### `FExpr`

If a function expression f() is not already marked as a method call, then we know there is no parent of this node that is a MethodCall. Then if there is a function with the name f in the class it is being called in, then we must convert it to a MethodCall where the object of the MethodCall is this.

### `FunctionDecl`, `FunctionDefn`, `FExpr`

Methods all take the object as a hidden first argument. If the FunctionDecl is a method we add this as the first argument.

We do not need to do this for FunctionDefn since this is encapsulated in the FunctionDecl that belongs to the FunctionDefn.

For all FExprs that are MethodCalls, qualified or unqualified must add this as the first argument.

### `Var`

We must lookup to see if the Var is a field and if it is we wrap it in a FieldAccess

### `SimpleDecl`, `ExtendedDecl`

When we see a decl we must add a new binding in the variable context. This is needed in case there is anything that should shadow the scope of a field. I have not handled this shadowing with function arguments. I wonder if the Var part of the FunctionDecl adds this to the context actually.

# IR Generation
We introduce three new changes to the IR:

1. Addition of `IRClassDefn`
2. Add list of `IRClassDefn` to `IRCompUnit`
3. Addition `IRDataArray`

The `IRClassDefn` class contains its methods in `IRFuncDefn` form and the `IRDataArray` which corresponds to the classes's static dispatch vector. The `IRCompUnit` hoists methods out of each class so assembly can be generated for each method. The intention is that method names all have unique encodings and will never conflict with regular function names. Therefore we can output them as normal functions and all code that is generated will call using the correct encoding. More details on the ABI conventions we came up with to accomplish this below.

The `IRDataArray` will need to be supported by the IRSimulator or assembly generation  Besides this, no changes are required below these changes.

## `ClassBody`

This class needs no conversion to IR because `ClassDefn` will do it all by entering the body.

## `ClassDecl`

This class needs no conversion to IR. At this point, all symbols are collected and type check has occurred. There is no use for `ClassDecl` anymore.

## `FunctionDefn`

FunctionDefns have a field isMethod. When we convert a method FunctionDefn to IR we must treat it differently than a normal function. Specifically, the ABI naming convention is different for functions and methods. We will discuss this convention below. But because of the difference between functions and methods, we do not convert to IR for methods and let the owning ClassDefn do the conversion. We refrain from allowing a method FunctionDefn from doing a conversion and discarding it because the conversion mutates state in the NodeToIRConverter and we would have to undo it later.

## `ClassDefn`

Like we just said above, `ClassDefn` is responsible for converting the `FunctionDefns` to IR using the correct method naming convention. It is also responsible for generating a dispatch vector for the class. This generation is quite involved and we will discuss in depth below.

## `FieldAccess`

This node converts itself by looking up the index of the field in the object representation and getting the offset into that array.

## `MethodCall`

We lookup the function in the dispatch vector and generate the same IR as a normal `FExpr` except that a `FExpr` uses a `IRName` but we are using a `IRMem` that contains the name from the DV lookup. We don't need to worry about passing the object to the function because the `ThisAugmenter` already took care of this for us`.

## `New`

We must allocate the new object and then call the initialization function. First we build the dispatch vector and get a pointer to it. Then we convert all the fields to IR. Then we create an array that contains this pointer followed by the fields. 

## `NodeToIRConverter`

We introduces a few new fields for bookkeeping:

1. `objectMethodEncodings` to store mangled function names. This allows us to store a mapping from method name to the actual ABI encoding for that function.T he encoding is the same as the encoding defined for a function or procedure in the Xi ABI except that the function name is proceeded by the class name and an underscore. See the section on DispatchVectorClassResolver for more information.
2. `classNameToNumFields` represents a mapping from class name to the number of fields that class has
3. `classNameToFieldNameToIndex` represents a mapping from class name to the index of the field in the object representation.

This class is also responsible for building the dispatch vector and getting the method symbol out of a dispatch vector.

# Dispatch Vectors

Dispatch Vectors were the biggest focus during this project. According to Professor Myers, there are [3 goals](https://www.cs.cornell.edu/courses/cs4120/2022sp/notes.html?id=objects) we had:

1. Method calls should be fast, since they are the core operation in the language
2. The space overhead per object should be low. We are less concerned with space overhead on a per-class basis, since the number of objects is typically much larger than the number of classes.
3. We would like to share inherited code across classes, since for a large class hierarchy, we might otherwise end up with many copies of the essentially the same code, leading to the dreaded “code bloat”.

A straightforward approach that has been used in dynamically typed OO languages like Smalltalk is to simply search for the right method definition. Each object has a hidden field that points to its class object; a method call `o.m()` is implemented by walking up the class hierarchy, starting from o's class, and stopping once an implementation for method m is found. This implementation allows sharing inherited code, so it avoids code bloat, but it is much slower than a simple function call.

For faster method dispatch, we need to precompute a data structure—a dispatch table. The key idea is to map different method names onto small integer indices, and to use those indices to look up the appropriate method code pointer in an array of code pointers. These arrays of code pointers go by various names. In C++, this array is called a vtbl, pronounced like “v-table”, standing for “virtual table". In other literature it is known as a selector table or dispatch vector.

So consider the following classes:

```
class A           { f(); g(); }
class B extends A { f(); h(); }
```

`A` and `B` should share the same call to `f`, `B` should inherit `g` from `A`, and `B` should also define `h`.

We can put `f` and `g` in indicies 0 and 1 respectivley for both dispatch vectors and `h` is put in index 2 for the dispatch vector of `B`. Then, if there is a call to `f`, `g`, or `h`, we can get the pointer to the correct function in `O(1)` time by generating IR that resembles the following

```
MOV dv_ptr, o // Get the pointer to the dispatch vector since it is at index 0 of the object o
MOV dv, [dv_ptr] // Get the dispatch vector
CALL [dv + index] // Call a function based on its index in the dispatch vector
```

We build two classes to assist with building dispatch vectors:

First we need to figure out when given some class `A`, and some function `f`, which classes `f` should we call. For example, if `f` is redefined by a subclass then we want to call the `f` of that subclass. But if `f` is defined by a super class we want to call `f` of that super class. This is all handled by the `DispatchVectorClassResolver`

The next thing we need is given all methods for a given class, what index do they go in in the dispatch vector. For example if `A` defines `f` and subclass `B` also defines `f` they must be at the same index. the `DispatchVectorIndexResolver` handles this.

Both of these classes have been well tested.

When generating the actual IR for the dispatch vectors, we tried a few different approaches. First, we converted the mangled function names into strings by representing them as an array of characters in the IR, and then tried to call the function by loading the string. We quickly realized that this would be difficult because the strings are longer often multiple characters and when we do `call [dv + index], we're really calling a function with the name corresponding the the first character in the mangled function name. We quickly scrapped this approach and tried to build them using static data.

In Xi, static data is represented by an `IRData` class. Out of the box, `IRData` is not a traditional `IRNode` (it does not extend `IRNode`. This means its children nodes are not `IRNode`s. Instead, it has a field `long[] data`. We tried to encode the dispatch vector as the concatenation of function names seperated by the null terminator. Again, we ran into the problem that when we do `call classADVLabel` it only reads the first word of data which corresponds to the first character.
Even worse, the interpreter tried to jump to the instruction at the address corresponding to that character instead of looking up that character in the function defn to address table and jumping there. I wrote a version of the interpreter that read until the next null terminator and then did the correct lookup. This got things to worked but it felt hacky because we could no longer write IR that jumped to an address -- it required we jumped to a function that was named as the string starting at that memory address until the next null terminator byte in memory.

We explored trying to put the memory address of the function in the IRData instead of the name. That way, we would only need to read one word and the interpeter would jump there as it wanted to do. The problem with this approach is that when we convert to the IR, we have no idea about the memory layout. The memory addresses are defined by the interpeter and would be different than the memory scheme if we went all the way to assembley. Additionally, there was no way to know about the memory layout of the simulator when converting to IR because it does not initialize its memory until after the IR is created.

We had two options: 

1. Modify `IRData` to allow us to have `IRName` or `IRConst` or `IRMem` nodes at each element in the data array instead of `long`s. 
2. Initialize a non-static dispatch vector for each object on construction so we could use `IRName`, `IRConst` and `IRMem` to represent the dispatch vector and offsets into it.

Both of these approaches would mean there were no changes to the `interpet` function of the interpeter, would allow the generated IR to work for both the `IRSimulator` and for assembley generation. The first approach however requried new classes to be created and modification of the `initialize` function of the `IRSimulator` to work with the new data class.

We opted for approach 2 for simplicity, although approach 2 is definetly the better approach. We discuss this more in future work section.

# Future Work

## Typechecking

This project skipped over the type checking phase so that more time could be spent focusing on the actual object oriented features. Obviously, this allows Xi programs that are not well typed and would likley crash or misbehave if these types of programs are used. Additionally, type checking would allow programs that involve more than one class. At the moment, it is impossible to figure out what class a variable `o` belongs to in situations such as `o.f()` and `o.x`. Currently, they are hardcoded to assume that all objects are of type `A`. Type checking would allow the compiler to identify what type a variable has and therefore make decisions based on class layout for any class type.

## Fields

All fields are private to a class. This means that if a class `A` has field `x` and class `B` entends `A`, then class `B` cannot refer to `x`. However, `B` can call methods of `A` and those methods may refer to `x`. Currently, subclasses do not allocate fields that belong to their parent classes. Therefore when a method of a super class is called, it has no way to get those fields. In the current implementation it will try and get the variable from the expected index which will really be an
undefined memory access.

In order to remedy this problem, we must allocate fields belonging to super classes at runtime since fields are not part of class declarations and are only part of class definitions by the Xi++ specification. I spoke with Professer Myers about this. C++ solves this problem by requiring public and private fields to be declared in the header files (the class declaration) so this information is known at compile time. Java on the otherhand does this at runtime by having a bytecode instruction that refers to the variables by name. In Xi, the runtime would probably need to do something similiar to the dispatch vector indexing scheme where all variables of a subtype are placed at a certain index in the object layout at runtime.

## Static Dispatch Vectors

As discussed above, much of the work that went into this project was with respect to the representation of dispatch vectors in the IR. I met with Professer Myers and he agreed that his intention is for these dispatch vectors to be static data but there is no way to represent them using the current version of `IRData` since `IRData` requires the physical address of functions but these are not determined until runtime. The solution is to create a different `IRData` class which can represent static data that is of the form `IRNode` instead of physical addresses. This would allow us to say the data is an array of `IRName`s which would be converted to the address of the correct function at runtime.

## Private Methods

The specification of Xi++ allows methods to be defined in class definitions that are not part of class declarations. The current implementation expects that this would result in a type checking error since the declaration and definition do not match. However, we could imagine that a class definition declares methods that it does not want to export. These types of functions could be represented in the object but not part of the dispatch vector. An alternative approach would be to store it in the dispatch vector but type checking would not permit `MethodCall`s outside the class to use it nor allow subclasses to use it.

## Evaluation
In addition to unit tests for Lexing, Parsing, `NodeToIRNodeConverter`, `ThisAugmenter` and dispatch vector tests we also have a series of integration tests that combine these features into the Xi/Xi++ compiler.

These integration tests include:
1. `class_equality.xi` - Checks if an object is equal to itself and if two objects are not equivalent to each other by using the `==` operator 
2. `class_field_access.xi` - Checks a series of class field accesses with inheritance
3. `class_method.xi` - A test for methods working for classes using a field
4. `field_access.xi` - A test for fields working for classes
5. `method_call.xi` - A test for methods working for classes without a field
6. `point.xi` - A comprehensive test using inheritence and various field and methods of classes
