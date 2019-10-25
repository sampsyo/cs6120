+++
title = "An Efficient Implementation of Self"

[extra]
bio = """
  [Rachit Nigam](https://rachitnigam.com) is a second year PhD student interested in
  programming languages & computer architecture. In his free time, he
  [subtweets](https://twitter.com/notypes/status/1170037148290080771) his advisor and [annoys tenured professors](https://twitter.com/natefoster/status/1074401015565291520).
"""
[[extra.authors]]
name = "Rachit Nigam"
link = "https://rachitnigam.com"
+++

[An Efficient Implementation of Self, a Dynamically-Typed Object-Oriented
Language Based on Prototypes][paper] presents techniques for runtime
compilation, now more commonly referred to as Just In Time (JIT) compilation,
for the [Self][] programming language.

> We have developed and implemented techniques that double the performance of
> the dynamically-typed object oriented languages.


## The Challenge

The term "dynamic language" is mostly commonly associated with modern scripting
languages like Python and JavaScript. Self, a much older language developed
at the famed Xerox PARC labs takes the philosophy of dynamism to a logical
extreme -- *everything* in self is a message to an object. This includes
Java-like method calls on object *as well as* control structures like loops
and conditionals.

For example, the Self condition `IfTrue:IfFalse:` is a method invocation on
the boolean object `true` and `false`. This means that a simple conditional:

```
if (x) 1 else 2
```

in Self parlance (written with JavaScript-esque syntax) looks like this:


```
x.If({ true: () => 1, false: () => 2 })
```

The runtime behavior of this program is invoking the method `IfTrue` on the
object `x`, which can be a `true` or `false` but is not required to be those,
and execute the "thunk" (a function with no argument) corresponding to the true
or the false branch.  Note that `x` is not required to be `true` or `false`.
Any Self object can define the `If` method and specify its conditional
behavior.

Considering this dynamism, a Self compiler must:

1. **Respect the semantics of method calls**. Restricting or specializing conditionals and primitives don't follow the spirit of the language.
2. **Provide interactive speed**. Recompiling after a small change is not an option because Self is meant for rapid exploration in a programming environment.
3. **Preserve stack traces**. Self supports extreme reflection and introspection. A programming environment must be debuggable.
4. **Generate fast code**. Interpreting the whole language is probably too slow.

## The Solution

While the paper goes into the nitty-gritty of object layouts and method
invocations, the essence of the paper can be summarized as:

>  When type information for an object is available, generate specialized code
>  and let inlining and compiler optimizations work their magic.

### Object Layout

Self programs use object prototypes to describe inheritance relationships.
Unlike classes, which have _constructors_ used to build _instances_, Self uses
prototypes, which act as _exemplars_ for other objects. Creating a new object
from a prototype is as simple as _cloning_ it and setting its parent pointer to
the prototype. Changing the common behavior of all clones is as simple as
changing the behavior of a prototype. Since method and field lookups traverse
the parent heirarchy, clones can also override methods and fields of their
prototypes.

A naive layout scheme for objects would copy all fields from a prototype and
end up wasting a lot of space describing potentially shared behaviors. The Self
compiler minimizes space usage of _clones_ derived from the same prototype by
using _clone families_. A cloned object only stores its modifiable in its
object and points to the clone family for its prototype. If the instance ever
overrides one of its methods, the creates a new clone family to preserve
semantics and propogate behavior changes to all clones of a prototype.

<img src="without-maps.png" alt="drawing" width="300"/>
<img src="with-maps.png" alt="drawing" width="300"/>


[paper]: https://dl-acm-org.proxy.library.cornell.edu/citation.cfm?id=74884
[self]: http://www.selflanguage.org/
