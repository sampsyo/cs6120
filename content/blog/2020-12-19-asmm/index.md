+++
title = "Automatic Static Memory Management in Bril"
[extra]
bio = """
  Patrick LaFontaine is a senior undergraduate student interested in programming language design and implementation. He also enjoys backpacking and attempting to bake the perfect loaf of bread.
"""
[[extra.authors]]
name = "Patrick LaFontaine"
+++

## Motivation: Compiler-led Garbage Collection

The [memory extension][memex] for the Bril programming language provides manually-managed memory for the allocation of arrays. Manual memory management, when the programmer is in charge of explicitly freeing their memory, provides full control to the programmer at the cost of exposing them to difficult-to-debug memory bugs. The common bugs are leaking memory by forgetting to call ```free```, attempting to ```free``` memory more than once, or attempting to use a memory location after it has been freed. A popular way to handle these issues is to have a run-time system in charge of deciding when to ```free``` memory. These run-time garbage collectors allow programmers to easily work with memory, but often at the cost of worse performance, poor latency with "stop the world" pauses or significant memory usage. One alternative to this is to put the compiler in charge of deciding when to free memory. This handles all of the memory management logic, at compile time, for the programmer, hence "Automatic Static Memory Management". The main downside of this approach is that this static analysis, like all compiler analyses, must be conservative which can limit the expressiveness of the programming language.

## A Region-Based type extension for Bril

This work, based on the [cyclone] language, seeks to bring region-based memory management to the Bril [memory extension][memex]. Each pointer to memory exists within the static scope of assigned memory called a region. Regions are hierarchical such that they can be nested within each other. Data that is allocated inside of a region can only be used within that region or inner-nested regions. When the scope of a region ends, all of the allocations that belong to that region are freed. In this way, the compiler statically rules out the previously stated, common memory bugs by enforcing when and for how long an allocation can be used for. For Bril, this comes in the form of an extension to the Bril Json representation and a static analysis tool that accepts this extension of Bril and converts it to standard Bril with manual memory management inserted into the program.

## Extension Design

The Type enum in the [bril_rs] library is modified as follows:

```rust
// I've removed the parts of this enum used in serializing/deserializing to json.
pub enum Type {
    Int,
    Bool,
    Float,
    Pointer(Box<Type>),

    /// New addition
    PointerRegions {
        pointer_type : Box<Type>,
        ownership : Option<Ownership>,
        region: String,
    },
    ///
}
```

The new addition is a pointer type which is annotated with the region it exists in and a concept of whether a pointer owns the memory it is pointing to. In Bril, every ```id```, ```load```, ```alloc```, and ```ptr_add``` can create a new pointer into memory. However, not all pointers are created equal. Pointer's
created by ```alloc``` are the original pointers into the zeroth index of memory and are treated like the "Owners" of that memory. It is these pointers that need to be freed when the scope of their region ends. Pointers created by ```id```, ```load```, and ```ptr_add``` are merely memory borrowers with views into heap memory. They still need to be statically checked that they aren't being used after the region scope ends but do not need a corresponding ```free``` instruction like memory owners do. What is important about this typing is that both pointer types can exist within a program depending on whether you want the region to be inferred or explicitly stated. Whether a pointer owns its memory is trivial to decide and it is not expected that programmers included this annotation. As will be touched on in the evaluation section, this opt-in use of explicit regions makes converting valid programs to this extension simple.

## Current Limitations on Programs

Outside of the type system additions, there are a few restriction put onto Bril Programs for them to be valid for this extension:

- Any variable that owns it's memory(returned from ```alloc```) can not be overwritten during the course of the program. This is to ensure that a program doesn't make multiple allocations using the same variable name but whereas the static analysis can only free the variables once.
- Currently, programmers cannot declare smaller regions within the function scope. Currently, each function has a region and each pointer argument has it's own region. The next step would be to allow the programmer to create their own scoped regions. This was set asider during the project proposal process but would be interesting further work.
- As a consequence of the first two, a programmer cannot currently allocate memory in a loop. The ideal way to do this would to have the loop body be one region so that at the end of each loop iteration any memory allocated during that iteration is freed. I attempted to infer small regions just for this use case and found it to be both difficult infer and confusing to use. It can be very unclear where the analysis has chosen to start and end a region which makes for a pour experience when the programmer can't tell when a memory location goes out of scope.
- Every Owning pointer must dominate the exit block(s) of it's region. In this way, you cannot conditionally allocate or free memory in a Bril Program. A given allocation of memory must exist in the same variable no matter which block it exits the region from. A borrowed pointer can be freely used as long as it is invalidated once the region scope ends.
- The programmer can specify when two arguments to a function should have the same region but they are unable to declare one region as nested within another. This would allow finer grain control over what input regions are acceptable for a function can instead the coarseness of, "Either they are the same region or they aren't".
- A pointer returned from a ```Call``` value instruction is assumed to own the memory it is returning. This is to make a clear distinction of who's job it is to free memory so that different functions don't double free the same memory at the end of their regions. This is more of an implementation limitation to make inference easier than a technical limitation.

## Implementation of the analysis

The analysis is broken up into 5 stages. First the Bril program is broken up into a control flow graph of basic blocks. This a straight forward phase which involves inserting extra labels which will show up in the resulting code. An inference pass is then run on each function to add in any region annotations that may not have been included. The function is given a region name if the programmer does not specify one. This is also when the program identifies the owner pointers from the borrowed pointers. From this point, regardless of user input, the control flow graph for each function is now fully annotated and consistent. Because this inference needs to be conservative, each argument that a program takes in is assumed to be a part of its own region. This is to avoid some kind of leakage bug where a pointer of short lived memory is stored in a long living array and it ends up leaving a gap after it is free.

Once all of the pointer types are fully annotated, the program can be quickly checked for some of the current limitations. For instance, when you call ```alloc```, we can check that all of the exit blocks that can be reached from this block are domination by the current block. This is to ensure when we come through with the next phase, any call to ```alloc``` is valid and needs to be free at each of the exit blocks dominated by this. We go through one more time and remove all of the region annotations to bring the Bril extension back to standard Bril code.

## Evaluation

Surprisingly, the inference part of this tool is strong enough that all four of the current benchmarks that use the memory extension can be converted to this extension of Bril merely by removing the ```free``` calls. The inference pass is able to annotate the rest in these case, ```eigth-queens.bril, fib.bril. mat-mul.bril, and sieve.bril```. The other benchmarks in the current suite do not use the memory extension but they can still be handed to and returned from this pass with no change. Three out of four of these benchmarks have the exact same total dynamic instruction count between the current benchmark and running this bril extension benchmark through the analysis. ```Sieve.bril``` was slower but this was due to the reconstruction of the program from the control graph being suboptimal and is not a performance issue with the extension itself.

With the exception of the ```mat-mul.bril``` benchmark, these benchmarks are not a very difficult evaluation. It may just be that full flexibility in memory management is hard to keep track of anyways so users are already restricting their use and these benchmarks are representative of some standard use. To compensate for this, I've gone and included the interpreter memory test cases to get fuller coverage of some of the edge cases that could occur. For instance, ```alloc_many.bril``` is a go example of a program that would fail to check in this bril extension.

Alloc_many.bril

```
@main {
  v: int = const 1;
  max: int = const 1000000;
  count: int = const 0;
.lbl:
  count: int = add count v;
  p: ptr<int> = alloc v;
  free p;
  loop: bool = ge count max;
  br loop .end .lbl;
.end:
  print count;
}

```

Notice how this program loops over the loop body some `max` number of times. Each time, the program allocates and frees some memory. Because the same variable destination is being used each time, this would not check. The allocation would need to be moved to the loop header and either freed immediately or at block `.end`. If there were user defined regions that you could write something like:

Alloc_many.bril.asmm

```
@main {
  v: int = const 1;
  max: int = const 1000000;
  count: int = const 0;
.lbl:
  start_region "loop_body"
  count: int = add count v;
  p: ptr<int> = alloc v;
  loop: bool = ge count max;
  end_region "loop_body"
  br loop .end .lbl;
.end:
  print count;
}
```

### Correctness

The correctness argument comes in two parts. First by using all of the memory tests/benchmarks, this bril extension will have as much coverage/correctness as the interpreter itself. The more logical argument some from the limitations that have been enforced that remove complex control flow like branches and loops for ```alloc``` calls. This means that you can look at the path of the dominator tree from the root to exit leaf nodes to find all of the owner pointers created for this region. Also by not including user created regions, this simplifies the region hierarchy with the current function at the bottom and each of it's arguments has a region different from the function one that is above it in the hierarchy. Another important part around function calls is that function arguments are always of a borrowed pointer type and the return pointer is always an owned pointer type. Therefore, you cannot duplicate a function pointer on accident and try ot free it twice. This gives a clear divide, it is the callers job to do the memory management for any pointer it gives or receives from the callee. It is a bit unclear if a programmer could give purposely wrong annotations to violate the soundness of this transformation. 

### Areas of Improvement

The two significant improvements that could be made are to allow the explicit creation of regions and a more expressive region hierarchy. Both of these were mentioned in the limitations section. Explicit region creation is especially important to allow programmers to manage branching memory allocations.

## Conclusions

[memex]:   https://capra.cs.cornell.edu/bril/lang/memory.html
[bril_rs]: https://capra.cs.cornell.edu/bril/tools/rust.html
[cyclone]: https://www.cs.umd.edu/projects/cyclone/papers/cyclone-regions.pdf