+++
title = "Adding OOP to the Gator Shading Language"
[extra]
bio = """
Evan Adler is a Masters of Engineering student at Cornell interested in compilers and programming languages. 
"""
[[extra.authors]]
name = "Evan Adler"
latex = true
+++

## Background

[Gator][gator_link] is a language created as a CU Capra project which compiles to GLSL. Its main purpose is to rule out subtle geometric bugs through a strong type system. I had recently added structs to the language to group related variables together and facilitate readable and maintainable source code. However, another reason that advanced programming language features are desired for Gator right now is that other target languages, aside from GLSL, are being considered. For example, making Gator an object oriented programming language will help Gator compile into C#. For this project, I compile object oriented Gator into GLSL.

## Main Design Decisions

Since GLSL has structs and functions, I thought a reasonable way to implement classes would be to strip out all the fields and place them into structs, and implement methods as functions which consume these "field structs".

I decided to implement single inheritance. In addition, all fields and methods have one of three access modifiers: public, private, or protected.

Since the target language, GLSL, does not contain function pointers, Gator cannot support dynamic dispatch. Therefore, valid Gator classes cannot define a method which is already defined by a parent class. At the moment, this is actually not enforced by the compiler.

## Constructor Fun

I originally designed constructors to work as follows. It should be noted that some of the syntax can be improved, particularly for constructors. CS 6120 is not about parsing ;)

```
class dog : animal {
    private float bark_volume;

    public dog init(float bv, int x, int y) {
        super(x, y);
        this.bark_volume = bv;
    }
};
```

My initial idea was to implement this by creating an underlying GLSL dog struct with default initial values. Then, I would set the `bark_volume` field to `bv` and the implicit `super` field to the result of calling the constructor for `animal`. This did not work because there was not a reasonable way to populate the fields with default values. For integers or floats, I could of course use zero, but for fields which are other objects for example, this approach becomes difficult.

The redesigned constructors work as follows. If the class has a parent, the first line must still be a call to `super`, but now the last line of the constructor must be a call to `self`, with the fields provided in the order they are declared in the class.

```
class dog : animal {
    private float bark_volume;

    public dog init(float bv, int x, int y) {
        super(x, y);
        self(bv);
    }
};
```

This way the underlying struct can be initialized at the end, when all field values are known, including the implicit super struct.

The above example indirectly calls the constructor for `animal`, but constructors can also be directly called as follows.

```
dog quiet_dog_factory(int x, int y) {
   return new_dog(0.25, x, y);
}
```

Idiotically, I realized when I performed the evaluation that there was a way to use my first constructor approach. Apparently, although GLSL doesn't provide default constructors for structs, you can simply declare a struct and then set the fields, without ever using a GLSL structure constructor.

## Sample Output

Sample output can be seen by invoking the gator compiler on the testing programs I made [here][test_link].

## Evaluation

This project was not inherently conducive to being evaluated quantitatively, but I wanted to see if the new OOP features could be added to real graphics programs without significantly affecting Gator compilation time, GLSL execution time, and GLSL code size.

Since the main purpose for OOP in Gator was for use with future target programming languages, it would be a bit of a waste of time to liberally sprinkle OOP features into existing Gator programs which target GLSL. GLSL is also not particularly suited for complex uses of OOP anyway. Since I was a bit rushed during the evaluation, what I ended up doing was taking the two Gator programs, a Phong lighting model and a microfacet lighting model, which I had previously added structs to, and just converting the structs into classes.

I measured compile times by using the linux `time` command while invoking the Gator compiler (taking the average of ten compilations). I considered modifying the JavaScript that drives the GLSL to print timestamps in order to measure GLSL execution time. However, I didn't bother comparing execution times because I knew the run-time behavior would practically be identical to using structs for the basic use cases I created. I also compared the byte counts of the GLSL output with and without using classes.

For Phong, I converted two structs into classes, one of which used a constructor, and the other didn't (it instead relies on a leaky abstraction in which it can interact with the object through its underlying struct. This hacky behavior should probably be disallowed later, but I found it convenient).

For Microfacet, I converted one struct into a class, and this did not require adding a constructor.

#### Without OOP:

| Program | Average Compilation Time (ms)| GLSL Output Size (bytes)
|---|---|---
| Phong Fragment Shader | 138 | 4773
| Microfacet Fragment Shader | 166| 7590

#### With OOP:

| Program | Average Compilation Time (ms)| GLSL Output Size (bytes)
|---|---|---
| Phong Fragment Shader | 141| 4831
| Microfacet Fragment Shader | 170 | 7590

As expected, compile times and code sizes did not significantly deteriorate in this experiment. It was interesting to observe the slight increase in compilation time, but this increase is clearly relatively negligible. The reason that microfacet's GLSL code size remains unchanged is because since it didn't use a constructor, the class just degenerated into being a struct.

## Future Work

I had originally wanted to implement subtype and parametric polymorphism. Parametric polymorphism should be easy, since Gator already supports it everywhere else (e.g., for functions). It would require adding parameterized structs, which should be supported anyway. In case the Java Generic style (the existing style in Gator) ends up being problematic, this could also be done like C++ templates, which would improve performance anyway. To support subtype polymorphism, the type system would need to be adjusted such that subclasses are also subtypes. In addition, there would need to be a way to dynamically search the object tree for fields. This is not hard to do, but it may be hard to do efficiently.

[gator_link]: https://github.com/cucapra/gator/
[test_link]: https://github.com/evanbenadler/linguine/tree/oop/test/oop