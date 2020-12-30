+++
title = "Adding OOP to The Gator Shading Language"
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

Below are some test Gator programs and their corresponding GLSL output.

#### Gator:
```
class d {
  public int a;

  public d init(int a) {
    self(a);
  }
};

class c : d {
  public int b;

  public c init(int a, int b) {
    super(a);
    self(b);
  }
  public int get_a() {
    return this.a;
  }
  public void set_a(int a) {
    this.a = a;
  }
  public int get_b() {
    return this.b;
  }
  public void set_b(int b) {
    this.b = b;
  }
};

void main(){
  c my_c = new_c(1, 2);
  int x1 = my_c.a;
  my_c.a = x1;
  int x2 = my_c.get_a();
  my_c.set_a(x2);
  int x3 = my_c.get_b();
  my_c.set_b(x3);
}
```

#### GLSL:
```
struct d { int a; };

d new_d( int a){return d(a);}


struct c { d super; int b; };

c new_c( int a,  int b){
    d super = new_d(a);
    return c(super, b);
}

int class_method_c_get_a( c self){return self.super.a;}

void class_method_c_set_a( c self,  int a){self.super.a = a;}

int class_method_c_get_b( c self){return self.b;}

void class_method_c_set_b( c self,  int b){self.b = b;}

void main(){
    c my_c = new_c(1, 2);
    int x1 = my_c.super.a;
    my_c.super.a = x1;
    int x2 = class_method_c_get_a(my_c);
    class_method_c_set_a(my_c, x2);
    int x3 = class_method_c_get_b(my_c);
    class_method_c_set_b(my_c, x3);
}
```

#### Gator:
```
class c1 {
  public int a;
  public c1 init(int x) {
    self(x);
  }
};
class c2 : c1 {
  public int b;
  public c2 init(int x1, int x2) {
    super(x1);
    self(x2);
  }
};
class c3 : c2 {
  public int c;
  public c3 init(int x1, int x2, int x3) {
    super(x1, x2);
    self(x3);
  }
};
class c4 : c3 {
  public int d;
  public c4 init(int x1, int x2, int x3, int x4) {
    super(x1, x2, x3);
    self(x4);
  }
};
class c5 : c4 {
  public int e;
  public c5 init(int x1, int x2, int x3, int x4, int x5) {
    super(x1, x2, x3, x4);
    self(x5);
  }
};

void main() {
  c5 c = new_c5(1,2,3,4,5);
  int x = 0;
  x = c.a;
  x = c.b;
  x = c.c;
  x = c.d;
  x = c.e;
}
```

#### GLSL:
```
struct c1 { int a; };

c1 new_c1( int x){
    return c1(x);
}

struct c2 { c1 super; int b; };

c2 new_c2( int x1,  int x2){
    c1 super = new_c1(x1);
    return c2(super, x2);
}

struct c3 { c2 super; int c; };

c3 new_c3( int x1,  int x2,  int x3){
    c2 super = new_c2(x1, x2);
    return c3(super, x3);
}

struct c4 { c3 super; int d; };

c4 new_c4( int x1,  int x2,  int x3,  int x4){
    c3 super = new_c3(x1, x2, x3);
    return c4(super, x4);
}

struct c5 { c4 super; int e; };

c5 new_c5( int x1,  int x2,  int x3,  int x4,  int x5){
    c4 super = new_c4(x1, x2, x3, x4);
    return c5(super, x5);
}

void main(){
    c5 c = new_c5(1, 2, 3, 4, 5);
    int x = 0;
    x = c.super.super.super.super.a;
    x = c.super.super.super.b;
    x = c.super.super.c;
    x = c.super.d;
    x = c.e;
}
```

#### Gator:
```
class human {
  public int x;
  public int y;
  public human init(int a, int b){
    self(a,b);
  }
};

class dog {
  public human owner;
  public dog init(human h) {
    self(h);
  }
};

void main() {
  human h = new_human(5,6);
  dog d = new_dog(h);
  int t = d.owner.x;
  d.owner.y = t;
}
```

#### GLSL:
```
struct human { int x; int y; };

human new_human( int a,  int b){return human(a, b);}

struct dog { human owner; };

dog new_dog( human h){return dog(h);}

void main(){
    human h = new_human(5, 6);
    dog d = new_dog(h);
    int t = d.owner.x;
    d.owner.y = t;
}
```

## Evaluation

This project was not inherently conducive to being evaluated quantitatively, but I wanted to see if the new OOP features could be added to real graphics programs without significantly affecting Gator compilation time, GLSL execution time, and GLSL code size.

Since the main purpose for OOP in Gator was for use with future target programming languages, it would be a bit of a waste of time to liberally sprinkle OOP features into existing Gator programs which target GLSL. GLSL is also not particularly suited for complex uses of OOP anyway. Since I was a bit rushed during the evaluation, what I ended up doing was taking the two Gator programs, a phong lighting model and a microfacet lighting model, which I had previously added structs to, and just converting the structs into classes.

I measured compile times by using the linux `time` command while invoking the Gator compiler (taking the average of ten compilations). I considered modifying the Javascript that drives the GLSL to print timestamps in order to measure GLSL execution time. However, I didn't bother comparing execution times because I knew the run-time behavior would practically be identical to using structs for the basic use cases I created. I also compared the byte counts of the GLSL output with and without using classes.

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