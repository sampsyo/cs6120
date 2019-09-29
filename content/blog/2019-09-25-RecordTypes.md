+++
title = "Record Types!"
[[extra.authors]]
name = "Katy Voor"
link = "https://github.com/kavoor"  
[[extra.authors]]
name = "Henry Liu"
link = "https://www.linkedin.com/in/liuhenry4428/"  
+++

The goal was to design and implement record types (aka structs). We decided on immutable record types using a nominal type system. We initially planned to implement record type declaration (that are named), record type instantiation, and record type accessing, but later decided to additionally implement _with statements_ to improve usability. The following code will be provided in human-readable bril, though we do update the `bril2json` tool to allow translations into the canonical json form. 

### To declare a record type
    type <record type name>  = 
        {<field1 name> : <field1 type> ; <field2 name> : <field2 type> ; … };

Where
`type` is a new keyword,
`<record type name>` is an identifier ,
`<field# name>` is an identifier, and
`<field# type>` is a type name, which may be either a primitive type or a previously declared record type.

We decided on this format to mirror OCaml [record type declarations](https://v1.realworldocaml.org/v1/en/html/records.html). However, unlike OCaml, we disallow recursive record types (i.e. a record type that may contain itself) as that would require complicated recursive types as well as a notion of nullable references, which are outside of the scope of this project.

### Nominal vs Structural Typing
One of the main design decisions was whether we wanted to use nominal or structural typing to typecheck records.

```
type Dog = {age: int; isAsleep: bool}
type Person = {class: int; dog: Dog}
v0: int = const 3;
v1: bool = const false;
Milo: Dog = record {age: v0; isAsleep: v1}
v2: int = const 4120;
AndrewMyers: Person = record {class: v2; dog: Milo}
```

When defining a nested record, it is required to first define the nested record and assign it to a variable. Then, you can define the outer record with the field initialized to the variable holding the nested record. When type-checking these nested records, we need to look up the type of the outer record from our type environment, and step through the fields, one by one, comparing the signature with the type returned from the lookup of the initializing variable. 

Consider the case for checking the variable initializing a nested record, i.e., `Milo`. With nominal typing, if we expected a type, Dog, and we looked up the variable to have type ‘Dog’, we know this must have been previously typechecked when it was defined and added to our environment. Therefore it still must typecheck, due to immutability. This sort of _shallow type-checking_ falls out of nominal subtyping. 

Along with the ability to quickly verify any type, nominal subtyping allows us to reject a nested record's type if it does not match the signature’s record type name without recursive checks. In a structural typing world, the name of the type bound to the initializing variable is not enough to reject a type. If the declared type did not match, we would need to recursively check all of the fields of the value bound to our initializing variable, and compare these with the type signature recursively. This is much slower and does not take advantage of the variable-save model to avoid recomputation.



### Instantiation
To instantiate a new record with a previously declared record type, we use the following format:

    <variable name>  : <record type> = 
        record {<field1 name> : <field1 value>; <field2 name> : <field2 value>};
Where:
`<variable name>` is an identifier,
`<record type>` is a previous declared record type,
`<record>` is new keyword,
`<field# name>` is the field name used in the record type definition, and
`<field# value>` is an identifier for an existing variable matching the field type

We decided to introduce the `record` keyword to match the precedent of using `id` or `const` in front of an identifier or constant value. Note that the ordering of field name and value pairs do not matter, as long as they match with the definiton’s field names and types. We also only allow field values to be existing variables to match the semantics of current operations. The structure of this statement was designed to match record type declarations as much as possible. 


### Access
We use the dot operator to access a field of a record:

    <new record name> : <type> = <record> . <field>

Where:
`<new record name>` is a valid identifier that takes on the value of the indicated field, and
`<record>` is the name of an instance of a record with a field name `<field>` that has type `<type>`
Note that there is a space before and after the dot. This is strictly necessary as dot is a valid character for a variable name and we decided that it would be horrible for backwards compatibility if we changed variable naming rules.


We chose this format because using the dot operator to access fields is very common in modern programming languages.

### *With* Syntax
While immutable data structures allow you to more easily reason about how values flow through your program, these currently immutable records are cumbersome to change. 
For example, to change one field, you must recreate the entire record. While we do want a new record to be created, copying variables between two records is overly tedious. 
As such, we decided to implement a with syntax, similar to OCaml in which the user just needs to specify a record name as the base, and then the fields desired to be changed, along with the new value. This maintains immutability, without so much tedium.

We use the following syntax:

    <new record>: <record type> = 
        <old record> with {<field1 name>: <field1 value>; <field2 name>: <field2 value> … };

Where:
`<new record>` is the name of the new record,
`<record type>` is the same type as `<old record>`,
`with` is a new keyword,
`<field# name>` must match with a field in `<type>`, and
`<field# value>` must be a variable with a type that matches its field name.
Within the braces, the user may specify 0 to n field name and value pairs, where n is the total number of fields in the record type.

This syntax was designed to have a similar format as record instantiation. 

### Immutability
Immutable data structures are easier to reason about when developing compiler optimizations. However, forcing developers to recreate the entire record every time they need to change a value poses usability challenges. We decided on sticking to immutability in part for optimizations' sake. The ability to make guarantees that a value will not change lends itself well to constant-folding, even after a function call that takes in your record as an argument. 

### Evaluation
To evaluate our implementation, we need to check that our implementation works as specified as a language feature, and that this language features satisfies the goal of extending Bril's ability to compile higher-level languages with record data types.

One of the goals of record types was to allow Bril to logically group and use related data points. This feature is useful when compiling higher-level languages that utilize a similar data structure like records in OCaml or structs in C into Bril. 

To measure success of this goal, we considered whether the functionality added from these new types increases Bril's ability to compile higher-level languages with record type data structures.

We implemented immutable nominal record types. This implementation includes record declaration, instantiation, and access. Decisions about which operations to include in our design were influenced by the operations available on record data types in higher-level languages. To test this functionality, we created a suite of tests that covered each of these operations, as well as combinations of these operations.

The primary operation not supported by our record type specification that is supported by some higher-level language record types is mutation. While our record types do not support mutation as discussed, this does not significantly hinder Bril's ability to compile higher-level languages that support this feature. 

Consider the following C code that declares a struct and provides a method to update.
```
struct Person {
   int age;
   bool isAsleep;
};

void updateAge(Person p){
    p->age = p->age + 1;
}

int main(void) {
    Person Henry = { 20, false };
    Henry.age += 1;
    return 0;
}
```
Compiling this program to Bril may look something like this, assuming we ignore function calls.
```
type Person = {age: int; isAsleep: bool};
v0: int = const 20;
v1: bool = const false;
Henry: Person = record {age: v0; isAsleep: v1};
v2: int = const 21;
Henry: Person = Henry with {age: v2};
```
As shown in this example, mutable record types can be compiled into immutable records in Bril without significant effort. Therefore, lack of this operation does not compromise this goal.

Finally, to evaluate record types as a language feature in Bril, we consider how this functionality translates to Bril.
We found that creating new records was a tedious process if the record was large, so we implemented *with* statements in addition to the features mentioned above for situations where one wanted to duplicate a record with a few changes. It should be noted that it is bad form to use a with statement with no fields because that would be identical to referencing the old record with `… = id oldRecordName`.

We were successful in this aspect as creating a new record from an existing one with this syntax is more concise and easier to reason about than copying over every field. This is an advantage in an IR as we can logically think about a single _with statement_ and a sequence of functionally equivalent statements that copy variables in the same way.

Consider the blocks of code below. These programs duplicate a record with one field changed.
Here we show what this would look like without _with statements_. 

```
type Person = {age: int; isHappy: bool};
v0: int = const 21;
v1: bool = const false;
Henry: Person = record {age: v0; isHappy: v1};
v2: bool = const true;
v3: int = Henry . age;
AwakeHenry: Person = record {age: v3; isAsleep: v2};
```
Here we use the with syntax.
```
type Person = {age: int; isAsleep: bool};
v0: int = const 21;
v1: bool = const false;
Henry: Person = record {age: v0; isAsleep: v1};
v2: bool = const true;
AwakeHenry: Person = Henry with {isAsleep: v2};
```
It is worth noting that the size of code required to duplicate a record without _with statements_ scales linearly with the size of the record. In contrast, the size of code required to duplicate a record with _with statements_ does not change as the size of the record increases. Therefore, record types are successsful as a language feature as they integrate well with current syntax and do not impose unneccesary code bloat. 

Overall, record types implement the specified record operations, and increase Bril's ability to compiler higher-level languages. 

### Notable Challenges
The design of records type went through multiple iterations before we were able to arrive at a specification that we felt was well-defined.
When making these decisions, one of the primary tradeoffs we identified was usability and ability to optimize.

We wanted records to be easy to use to store data, but with immutability to be able to better reason about what values variables could be at any point in our program. In addition, since these are value types, changing one field will only change the field of the copy of the record you are modifying. Therefore, the same functionality can be achieved, but in a safer way, with _with syntax_. These sort of design decisions and consideration of use cases was a challenge as we looked at other prior art to get ideas, but ultimately wanted these types to be useful in Bril. 

A different challenge of this project was probably debugging the typescript interpreter as it was very difficult to trace errors as the source typescript gets compiled into a separate javascript file. Debugging the parser in briltxt was not too bad as the new statement formats were pretty straightforward and we did not have to modify existing semantics. 
