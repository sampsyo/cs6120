+++
title = """Bril extension for algebraic data types"""
[extra]
bio = """
Will is a senior undergraduate student interested in programming language implementation. He also enjoys autonomous robotics and music peformance."""
[[extra.authors]]
name = "Will Smith"
link = "https://github.com/Calsign/"
+++

The result of my final project was the addition of support for product
types (i.e., tuples) and sum types (i.e., variants) to Bril. I
completed each of the following:

 - Defined four new instructions: `pack`, `unpack`, `construct`, and
   `destruct`
 - Added support for these instructions to `brili`
 - Added support for these instructions to `bril2txt`/`bril2json`
 - Created a type checker with support for these new instructions;
   written in OCaml, located in
   [`algebraic-types/type-check`](https://github.com/Calsign/bril/tree/algebraic-types/algebraic-types/type-check)
 - Created a Rust compiler for turning a subset of Rust code into
   Bril; written in Rust, located in
   [`bril-rs/src/main.rs`](https://github.com/Calsign/bril/blob/algebraic-types/bril-rs/src/main.rs)

Additionally, I added support for the new instructions to the
TypeScript, Ocaml, and Rust libraries as required to do the above.

The code is available on the [`algebraic-types`
branch](https://github.com/Calsign/bril/tree/algebraic-types) of my
fork of the Bril repo.

The remainder of this blog post is devoted to documentation, design
choices, and evaluation.

## Design: algebraic types

A product type is a type that acts as a block containing zero or more
other types (a Cartesian product). Commonly, the two case is called
a pair and the zero case is called unit.

A sum type is a type that at any given time may contain any of one or
more types, where which type a given value has may be inspected at
runtime (a tagged union). Each type in a sum type is referred to as a
constructor. Note that the zero case is uninstantiable, so we do not
allow it.

Product and sum types are represented in Bril as new kinds of
parameterized types. In JSON, they are as follows:

 - Product: `{"product": [t1, ..., tn]}`
 - Sum: `{"sum": [t1, ..., tn]}`

In text, they are as follows:

 - Product: `product<t1, ..., tn>`
 - Sum: `sum<t1, ..., tn>`

I chose to make the sum types anonymous, as recommended by
Prof. Sampson. Many high-level languages with sum types, like OCaml,
have named variants with named constructors rather than anonymous sum
types with anonymous constuctors, but adding named types directly to
Bril would have been perhaps unnecessary and unnecessarily
complicated.

I originally planned to support recursive types, like the following
OCaml definition of a list type:

```ocaml
type intlist =
  | Nil
  | Cons of int * intlist
```

Unfortunately, it is not clear to me how to easily support recursive
types without named sum types, let alone potential other complications
with recursive types that I did not consider.

Here are some examples of types that are supported by my extension:

 - `product<>`
 - `product<int, bool>`
 - `sum<int, product<>>`
 - `sum<product<int, int>, bool>`
 - `product<sum<int, bool>, int>`

Note that while sum types with duplicate constructors (e.g. `sum<int,
int>`) can be declared with the above type definitions, as we will
see, it is not possible to differentiate between different
constructors of the same type at runtime, so such sum types are not
useful.

I considered adding a subtyping rule for sum types that would permit
using, for example, `sum<int, bool>` in places that require `sum<int,
bool, product<>>`. I decided to leave this out because it seemed
excessive for an IR and difficult to implement.


## Design: new instructions

The basic idea was to add a pair of new instructions for both product
and sum types; one for creating values of the type, and one for
examining those values.

### Product types

#### `pack`

The `pack` instruction is used to create a tuple. It takes in an
arbitrary number of values to pack into the tuple.

The destination type of this instruction must be a product type with
an arity equal to the number of arguments, and if the destination type
is `product<t1, ..., tn>`, then the `i`th argument must be of type
`ti`.

The JSON format of the `pack` instruction is a follows:

```json
{
    "op": "pack",
    "args": ["a1", ..., "an"],
    "dest": "d",
    "type": {"product": [t1, ..., tn]}
}
```

The textual format of the `pack` instruction is as follows:

```
d: product<t1, ..., tn> = pack a1 ... an
```

#### `unpack`

Conversely, the `unpack` instruction is used to extract an individual
value from a product-typed value. It takes two arguments: a variable
(which must be a tuple), and the (zero-indexed) index of the value
from the tuple to extract.

Given that the input tuple has type `product<t1, ..., tn>` and the
index is `i`, the destination must have type `ti`.

The JSON format of the `unpack` instruction is as follows:

```json
{
    "op": "unpack",
    "args": ["a", "i"],
    "dest": "d",
    "type": "t"
}
```

The text format of the `unpack` instruction is as follows:

```
d: t = unpack a i
```

#### Example

Here is a sample Bril program that uses tuples:

```
@main() {
  one: int = const 1;
  two: int = const 2;
  tuple: product<int, int> = pack one two;
  one_again = unpack tuple 0;
}
```

### Sum types

#### `construct`

The `construct` instruction is used to construct sum types. It takes
as its sole argument the value to use. The semantics of the
instruction dictate that constructor to use is determined
automatically by examining the runtime type of the argument.

The destination must have a sum type and the type of the argument must
correspond to one of the constructors of that sum type.

The JSON format of the `construct` instruction is as follows:

```json
{
    "op": "construct",
    "args": ["a"],
    "dest": "d",
    "type": {"sum": [t1, ..., tn]}
}
```

The text format of the `construct` instruction is as follows:

```
d: sum<t1, ..., tn> = construct a
```

In theory, it could have been possible to omit `construct` entirely if
there were a sum type subtyping rule in place, but such a rule would
probably have made things more, not less, confusing.

#### `destruct`

The `destruct` instruction is used to execute different code based on
the constructor used to create a sum-typed value. It takes a sum-typed
value as an argument and a number of labels, one for each constructor
in the sum type.

The destination of the instruction is a variable name with a sum
type. The semantics of the instruction dictate that the control flow
jumps to the correct branch based on the argument and that the
destination value gets assigned the value within the constructor.

The destination will in fact not have the same sum type as the
argument; it will have the type carried by one of the constructors,
which is different for each label. This is confusing because the
destination is annotated with a type that it cannot have, but I chose
this approach because the Bril interpreter needs to know the available
constructors in order to select the correct label to jump to.

The JSON format of the `destruct` instruction is as follows:

```json
{
    "op": "destruct",
    "args": ["a"],
    "labels": ["l1", ..., "ln"],
    "dest": "d",
    "type": {"product": [t1, ..., tn]}
}
```

The text format of the `destruct` instruction is as follows:

```
d: sum<t1, ..., tn> = destruct v .l1 ... .ln
```

A common use case is some way of representing an empty, or unit,
value. For this it is recommended to use the empty product type,
i.e. `product<>`. For example, to represent an optional `int`, one
could use the following type: `sum<int, product<>>`.

I debated whether to even include a powerful instruction like
`destruct` at all. I could have instead offered an instruction to get
the index of the constructor used for a value and another instruction
for casting a sum type down to one of the constructor-carried
types, using a series of branch instructions to replicate the behavior
of `destruct`.

The reason that I opted for the `destruct` approach is that it makes
it possible, and not terribly difficult, to type-check the usage of
sum types in a Bril program. With the alternative approach, it is not
clear to me that it is even possible to perform type-checking of
things like the casting instruction described above. `destruct` does
seem rather large and clunky for an IR, especially compared to the
other instructions (it's like a whole bunch of `br` instructions in
one!), but I think the ability to perform type-checking makes it
worthwhile.

#### Example

Here is a sample Bril program that uses sum types:

```
@main() {
  one: int = const 1;
  sum: sum<int, bool> = construct one
  val: sum<int, bool> = destruct sum .int .bool
.int:
  res: int = add val one
  jmp .end
.bool:
  res: int = const -1
.end
}
```


## Implementation: type checker

The type checker is an [OCaml
program](https://github.com/Calsign/bril/tree/algebraic-types/algebraic-types/type-check)
that accepts a JSON-formatted Bril program on standard input. It exits
cleanly if the program is well-typed and crashes with a
(possibly-helpful) error message if it is not well-typed.

The program can be built with `make` in the
`algebraic-types/type-check` directory and run with `./main.byte` from
that directory.

The type checker does not support any extensions aside from algebraic
types, but it is fully functional with all of core Bril.

### Algorithm

The type checker algorithm is a dataflow analysis. I worked off of the
dataflow analysis
[framework](https://github.com/Calsign/cs6120_work/tree/master/lesson4)
that I made for [lesson
4](https://www.cs.cornell.edu/courses/cs6120/2020fa/lesson/4/).

The work set of the dataflow analysis is a mapping from variables to
types. The meet operator reverts to knowing nothing about a variable
in the event of conflicting types and the transfer function operates
straightforwardly based on the semantics of each instruction.

The tricky part is handling the `destruct` instruction. The analysis
works by storing a bit of extra information in the work set if the
previous instruction was a `destruct` and then assigning the correct
type to the variable based on the label that occurs next. This
approach seems reliable with only minimal changes needed to the
dataflow analysis framework.

The type checker does not support one label/basic block being the
target of different match constructors with different type
payloads. The type checker rejects all programs that attempt to do
this because it was deemed not worth adding unnecessary things to the
fancy `destruct` handling. In practice, real programs should never do
this because it is not a good idea, so I do not consider this to be a
problem.


## Implementation: Rust compiler

The Rust compiler is a [Rust
program](https://github.com/Calsign/bril/blob/algebraic-types/bril-rs/src/main.rs)
that accepts a Rust source file on standard input and outputs a
JSON-formatted Bril program on standard output that implements the
input Rust code. It supports a small but useful subset of the Rust
language syntax.

The program can be built with `cargo build` from the `bril-rs`
directory and run with `cargo run` from that directory.

Critically, the compiler makes no attempt to follow the semantics of
Rust; for example, there is no type checker or borrow checker. The
compiler just leverages Rust syntax to make it easier to write Bril
programs.

The compiler is akin to the [TypeScript
compiler](https://capra.cs.cornell.edu/bril/tools/ts2bril.html). I
decided to make a new compiler in Rust rather than expanding on the
existing TypeScript one because I wanted to compile a language that
had named variant types. (I also wanted to write the compiler in a
language that I preferred.)

The compiler uses the Rust [syn
package](https://docs.rs/syn/1.0.54/syn/) to parse the source code.

### Supported Rust features

The Rust compiler supports the following Rust types:

 - `i32`
 - `bool`
 - tuples
 - enum variants with arbitrary payloads

Note that the compiler does not support variants with multiple
constructors that have the same payload type. For example, here is an
enum that is not permitted:

```rust
enum Bad {
    FirstConstructor(i32),
    SecondConstructor(i32),
}
```

The reason for this restriction is that the underlying Bril
representation is an anonymous sum type and the interpreter can't tell
the difference between two constructors unless their payloads have
different types.

The Rust compiler supports the following Rust syntactic constructs:

 - Functions
 - Expressions
   - Constants (`i32`, `bool`)
   - Unary operators (`-`, `not`)
   - Arithmetic binary operators (`+`, `-`, `*`, `/`)
   - Comparison binary operators (`=`, `<`, `<=`, `>`, `>=`)
   - Logical binary operators (`&&`, `||`)
   - Tuple creation
   - Tuple access (e.g., `tuple.0`)
   - Constructors
   - If
   - While
   - Function calls
   - Match
 - Let bindings
 - Return
 - `println!` macro
 - Variant declarations

Expressions can be arbitrarily nested because the compiler performs a
primitive form of tiling, generating arbitrary temporary variables as
necessary. The resulting Bril code is very bloated as a result and
could benefit tremendously from some basic optimization passes.

Let bindings must have type annotations attached, for example:

```rust
let x: i32 = 42;
```

This restriction comes from the fact that all Bril value instructions
require a type. It would be possible to add type inference to the
compiler, but that would have been a lot more work.

Another effect of this restriction is that arbitrary sub-expressions
are not allowed anywhere that the type cannot be determined (only a
variable is permitted). This occurs with an expression that is being
matched on, so the compiler does not support matching on arbitrary
expressions. Note also that only single-level matching is supported
(no patterns beyond constructors). The matching support is essentially
a thin wrapper around the `destruct` instruction.

The compiler also does not support arbitrary expressions in `println!`
macros, but that is because `syn` does not fully parse macro
contents. Also note that `println!` macros do not use format strings;
they aren't actually Rust `println!` macros, behind the scenes they
just transform something like `println!(a, b, b)` into `print a b c`
in Bril.

### Sample

Here is a sample Rust program (`safe_division.rs`) that can be compiled to Bril:

```rust
fn main(x: i32, y: i32) {
    let res: Res = safe_divide(x, y);
    match res {
        Res::Res(res) => println!(res),
        Res::DivideByZero => {
            let pr: i32 = -1;
            println!(pr)
        }
    }
}

enum Res {
    Res(i32),
    DivideByZero,
}

fn safe_divide(x: i32, y: i32) -> Res {
    if y == 0 {
        return Res::DivideByZero;
    } else {
        return Res::Res(x / y);
    }
}

```


## Challenges

The design for the new instructions seemed fairly obvious to me, so I
formalized them but otherwise didn't spend much time working on them.

Adding `brili` support was also fairly straightforward, but I was
frequently frustrated by TypeScript because it was new to me and I
found some of the typing rules rather non-intuitive.

The type checker was also not terribly difficult, although it was a
bit tedious because I had to add rules for every Bril
instruction. Figuring out how to type-check `match` instructions
within the dataflow analysis framework was a bit challenging.

The Rust compiler was by far the most difficult and time-consuming
part of the project. First, I spent a fair amount of time trying to
figure out how to parse Rust source code before settling on the `syn`
package. Figuring out how to use the parsed AST was also an involved
process because it's a full AST for a "real" language, not just a toy
language like Bril. I also spent quite a bit of time figuring out how
to recursively tile expressions and how to keep track of variant
types. It's also worth mentioning that I'm still a little new to Rust,
so I was occasionally roadblocked by things like borrowing issues,
although I am definitely getting better.


## Evaluation

### Correctness testing

I created three sample Bril programs using algebraic types in the
[`algebraic-types/samples`](https://github.com/Calsign/bril/tree/algebraic-types/algebraic-types/samples)
directory as well as two sample Rust programs in the
[`algebraic-types/samples-rs`](https://github.com/Calsign/bril/tree/algebraic-types/algebraic-types/samples-rs)
directory.

I tested the correctness of the `brili` additions by running the
sample Bril programs and observing the output. The samples provide
complete coverage of the added instructions, which isn't terribly
difficult because there are only four instructions.

I tested the correctness of the type checker by running it on some of
the provided Bril benchmarks as well as the Bril samples for testing
algebraic types. I also ran it on modified versions of these programs
that were designed to fail the type checker to verify that it failed
accordingly. I probably have not achieved complete coverage with
respect to the core Bril instructions, but I have tested extensively
with the algebraic type instructions and definitely have achieved
complete coverage there.

I tested the Rust compiler in several ways. I only tested it with two
sample programs, but the two programs together cover all of the
supported Rust syntax. I converted the output to the textual Bril
format and examined the code to verify its correctess; I then ran it
through the type checker to verify that the output programs have valid
types; and finally, I ran it through `brili` to ascertain that the
compiled programs output the expected values.

### Reflection

I think that using a powerful `match` instruction was the right
decision. Having the ability to perform type-checking made me a lot
more confident in the correctness of the Rust compiler.

I do think that it would have been preferable to make the `construct`
instruction require explicitly specifying the index of the constructor
to construct. This approach would have been better for a number of
reasons: it would have eliminated the dependency on runtime knowledge
of types; it would have made the Rust compiler's job easier; it would
have made the code more explicit and readable; and it would have made
it possible to use variants with multiple constructors that carry the
same payload type. When I realized this shortcoming, I didn't have
enough time to go back and change the project, unfortunately. That
being said, the current implementation is not horribly wrong, I think
it is just sub-optimal.
