+++
title = "A Class-based Object Extension to Bril"
[extra]
latex = true
[[extra.authors]]
name = "Yulun Yao"
+++

## About Bril

Bril is a low-level intermediate representation (IR) language which is:
- Typed
- Language agnostic
- With minimal core
- Instruction Oriented

The above charateristics have made Bril a good low-level IR, as it's clean and simple. Many CS 6120 students in the past semesters have attempted to extend Bril. Some notable examples include **Pointers and Heap Memory Management**, **Structs**, **VLIW support**. 

The extensions have allowed more interesting optimization to be made, as well as bridged the abstraction gaps in between higher level languages with Bril.

## OOP, and Bril??
Object-Oriented Programming is a popular programming model that organizes program around objects. It offers some notable expressibility, reusablity, and extensiblity, while not being super complicated. A lot of modern programming languages do support this paradigm. Although Bril is a low-level IR, providing some basic supports to object would make Bril way more expressive and interesting for people to play with, maybe as a tool for potential transpiler that converts a program in high-level language to Bril. This motivates us to add object representation in Bril. A central challenge of this project is that Bril is very limited in terms of its core language syntax, and we have very limited tools to build object upon Bril. We also want to remind ourselves that, Bril is, foundamentally low-level and agnostic to any higher-level language. Therefore, we don't want to add too much complexity to Bril to the extend that new abstractions would become a burden. We have decided to make this extension minimally evasive so that it does not interfere with any Bril's core syntax.

## What I've actually done?
In this project, I have designed a couple of new primitives, which provide higher-level abstractions. The whole implementation, is basically just a compiler pass which translates a Bril program with object-related syntax to a Bril program with only minimal/core syntax, so that it's completely modular and does not change the behavior of current Bril programs.

## A class-based implementation
A key question in the design is that whether to use a class-based object model or a prototype-based object model. In a class-based model, objects are pre-defined for their behaviors and types, meanwhile in a prototype-based model, objects are the primary entities, and objects pass all methods and values with each other through delegations. While a prototyped-based model is considered much simpler, I picked the class-based model since Bril is statically-typed and I don't want to add something that break the well-formedness of Bril. It also suits the current Bril toolings more and would get us rid of hectic dynamic typecheckings.

## An example
Let's start from an extended Bril program, which I would use to explain any new syntax and translation.

``` 
%rectangle {
  length: int;
  width: int:

  @init(input_l: int, input_w: int){
    lengthref: ref = dot this length;
    widthref: ref = dot this width;

    set lengthref input_l;
    set widthref input_w;
  }

  @area: int{
    lengthref: ref = dot this length;
    widthref: ref = dot this width;

    length: int = get lengthref;
    width: int = get widthref;

    prod: int = mul length width; 
    ret prod;
  }
}

@main {
  a: int = const 4;
  b: int = const 2;

  rect1: rectangle = new rectangle;
  initref: ref = dot rect1 init;
  call @initref a b;
  
  arearef: ref = dot rect1 area;
  area: int = call @arearef;
  
  print area;

  delete rect1;
}
```

This Bril program takes in two integers `a`, and `b`, uses them as `length` and `width` of a rectangle, and then outputs the `area` which is the `product` of the two variables. Compare to the world where we do not have classes and objects, this clearly has a lot of overhead in terms of extra instructions. However, we now clearly see that we are forming a rectangle with the given inputs, and we know we are computing the area of the rectagle we just formed.

So what the translated Bril program with only core syntax would look like?

```
@rectangle.init(this.length.ptr: ptr<int>, this.width.ptr: ptr<int>, input_l: int, input_w: int) {
  store this.length.ptr input_l;
  store this.length.ptr input_w;
}

@rectangle.area(this.length.ptr: ptr<int>, this.width.ptr: ptr<int>) {
  length: int = load this.length.ptr;
  width: int = load this.width.ptr;

  prod: int = mul length width; 
  ret prod;
}

@main {
  a: int = const 4;
  b: int = const 2;
  
  one: int = const 1;
  rect1.length.ptr: ptr<int> = alloc one;
  rect1.width.ptr: ptr<int> = alloc one;

  call @rectangle.init rect1.length.ptr rect1.width.ptr a b;

  area: int = call @rectangle.area rect1.length.ptr rect1.width.ptr;

  print area;

  free rect1.length.ptr;
  free rect1.width.ptr;
}
```

The translated program is actually shorter, given how much condensed the core Bril is.

## Syntax
Now let's look at the syntax in the example program.

### Class Definition / This
A class definition looks somewhat similar to a function definition, with three major differences:
1. We could now define functions in the scope of class.
2. We can define variables without taking any operations. (As we are not running anything.)
3. We now have `this` identifier to indicate fields/functions within the class.

Compared to the OOP implementation in a high-level language, there's currently no inheritance, no encapsulation (every field can be visited globally by the `dot` operator), no polymorphism (this abstraction is too high-level and would cause confusion, which makes no sense implementing in Bril). 

### Reference / `dot` / `get` / `set`
`Reference` is a new type which is analogous to the `Pointer` type, but with a higher level of abstraction as it deals with objects rather than heap memory. We use `dot` to obtain the reference to each fields of object, including any class functions. The `dot` operation takes in two arguments, which are object names and field names --- `dot rect1 area` just means `rect1.area`. A reference to a class function can be directly passed into `call` operation, while a reference to a field variable must be used with `get` and `set`, which are similar to `load` and `store` in pointer operations. I was very hestitated in adding `get` and `set` since it might make the syntax too complicated, however, I believe it's useful in following ways:
1. Just like `dot` operator, `get` and `set` are very common abstractions in object.
2. `get` and `set` could hide the underlying implementation of class (pointer operations) which we translate into.
3. `get` and `set` is independent of type and implementation.

### New / Delete
`new` allocates the space for an object with regard to a certain class. It also statically bound the object to that class. `delete` frees up the space for the object.

## Translation
None of the extended syntax is perserved into the translated Bril program. As I mentioned earlier, I am trying to be as being non-invasive as possible to keep Bril simple and low-level. Therefore, I did not change the Bril Interpreter even though it was very tempting to me. The implementation of this work is therefore a translation pass from extended syntax to the core Bril syntax.

The underlying implementation of all fields are pointers, as an object is stored in heap. My initial attempt was to follow a similar approach as `high level language -> LLVM`. It does not work well in this case, because the tools I have is more restrictive and limited. The pointers are strictly typed and are limited to certain pointer arithmetic. So that I was unable to put all the fields in a class into a consecutive block of heap. Instead, I have to allocate a block per field. Hence a `new` operation is essentially `alloc` on every fields, and `delete` is just `free` for every fields.

The most interesting object operation is `dot`, which does a lookup over the class definition, which is a map from field names to types. The types are always parameterized type in the form of `{"ptr": type}`, i.e. `ptr<type>`. When we obtain the function reference, it's also important that we pass all the fields needed in the object as additional arguments. This is also resolved when during the translation of `dot`.

The class definition is used for static analysis that provides the information to every steps in the translation, which are:
1. Read class information, extract class members.
2. Convert class function's signature to regular Bril function's.
3. Create a mapping in between instantiated objects and the actual pointers generated.
4. Convert all new/delete operations.
5. Convert all dot operations, which is also responsible for figuring out all arguments.
6. Clean-up all non-core-Bril syntax from the JSON representation.

## Difficulty
The main difficulty of this project is to design proper abstractions. The new abstractions need to be expressive. However, it cannot break the current Bril design and I should reuse current Bril syntax as much as possible. The actual implementation did not took me too long, but I spent quite some time figuring out all the design trade-offs and some difficult cases.

There are some scenerios which are probably unsolvable without a drastic change to the Bril interpreter, for example, having objects as arguments and return values would be a major challenge, as we would have to pass several pointers along, and we currently lack tools to do that. Another challenge is to track all the objects and references, similar to the problems raised in implementing pointers.

## Evaluation
I evaluated my design on several test-cases, based on correctness of the translation (qualitatively), overhead in terms of translation (quantitatively) and the extra instructions in the Bril program (quantitatively).

Here is my result:



## Future work
I would leave any further OOP features as future work, however, I would love to fix the scenerios I mentioned in the previous section first for the general correctness.