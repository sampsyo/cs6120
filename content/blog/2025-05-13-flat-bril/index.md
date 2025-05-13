+++
title = "A Flattened Representation for Bril"
[[extra.authors]]
name = "Ernest Ng"
[[extra.authors]]
name = "Katherine Wu"
[[extra.authors]]
name = "Samuel Breckenridge"
+++

# A Flattened Representation for Bril

Typically, the way to implement an interpreter is to create an explicit AST data type, but this requires allocating AST nodes on the heap. Instead, as demonstrated in [this blogpost](https://www.cs.cornell.edu/~asampson/blog/flattening.html) by Adrian, we can flatten the AST into an array, i.e. pack all AST nodes into one single contiguous array, and refer to children in the AST using array indices (as opposed to pointers). This has enormous performance benefits!

Since Bril is based on commands rather than expressions, there are no ASTs to flatten. However, in the existing Rust interpreter, programs, functions and instructions are all represented using structs that contain pointers to the heap. For instance this is the representation of a value instruction:

```Rust
Value {
        args: Vec<String>,
        dest: String,
        funcs: Vec<String>,
        labels: Vec<String>,
        op: ValueOps,
        pos: Option<Position>,
        op_type: Type,
    }
```

`Vec`s and `String`s in Rust are both implemented using pointers, so the representation is clearly not flat even before considering that functions contain pointers to instructions and programs contain pointers to functions. Our goal was to adopt the same approach of flattening data structures and packing them into a single contiguous array for Bril. The resulting representation of Bril programs is then a flat file containing these arrays that can be directly mapped into memory and interpreted, analogous to the approach laid out in [another blogpost](https://www.cs.cornell.edu/~asampson/blog/flatgfa.html) by Adrian. This requires two main components:

1. Infrastructure to convert existing Bril JSON files to/from our flattened format
2. An alternate Bril interpreter that operates directly on the flattened data structure (as opposed to the existing one brili, which has to parse JSON)

Our complete implementation can be found at https://github.com/ngernest/flat-bril/tree/main

## Design

### Flat data structures

A key design decision was what the flattened data structures would look like. We clearly need similar fields to those of the original representation, but we need a strategy for flattening `Vec`s and `String`s and other non-flat types. Instead of storing variables, labels and function names as Strings and Vec<String>s we aggregate all the names referenced throughout a function into three contiguous arrays of bytes that are stored in our function representation. Then instead of using Strings, our instruction representation stores pairs of indices that can be used to extract the relevant name from the function-level contiguous array. One subtlety here is that if we want to use a Vec, we cannot just index into the array containing all the names, we would not know where the boundaries between each name are! We need to use an intermediary array that stores indices into the top-level name array.

For instance, in a function with three variable names v1, v2 and v3, the function-level variable array will look like v1v2v3. If an instruction has two arguments v1 and v2, our representation stores a pair of indices into an intermediate array that itself contains pairs of indices. A lookup would proceed as (0,1) → (0,0), (1,1) → v1, v2.

We applied these same ideas consistently across the original Bril representation to produce flattened versions of functions and instructions:

```Rust 
pub struct Instr {
    pub op: u32,
    pub label: Option<(u32, u32)>,
    pub dest: Option<(u32, u32)>,
    pub ty: Option<Type>,
    pub value: Option<BrilValue>,
    pub args: Option<(u32, u32)>, // Indirect: indexes into args_idxes_store
    pub instr_labels: Option<(u32, u32)>, // Indirect: indexes into labels_idxes_store
    pub funcs: Option<(u32, u32)>,
}
```

```Rust 
pub struct Function {
    pub func_name: Vec<u8>,
    pub func_args: Vec<FuncArg>,
    pub func_ret_ty: Option<Type>,
    pub var_store: Vec<u8>,
    pub args_idxes_store: Vec<(u32, u32)>, // Intermediate array: indexes into var_store
    pub labels_idxes_store: Vec<(u32, u32)>, // Intermediate array: indexes into labels_store
    pub labels_store: Vec<u8>,
    pub funcs_store: Vec<u8>,
    pub instrs: Vec<Instr>,
}
```

Although our function representation does contain Vec s to enable construction, these will be flattened to slices once the entire Function has been created

### A flat file format

Once we have flattened all of the functions and instructions in a Bril program, we are left with an array of functions, which is a full Bril program! We needed to come up with a way of writing this array to a file so that our interpreter could read the file and recover the flat data structures. To do so we first associated a table of contents with each function. As our flat function representation consists solely of arrays of bytes or arrays of indices into other arrays, what we need to record is the size of each array. We prepend this to every function, then as the table of contents has a fixed length it can be used to recover each of the function’s individual arrays. We confront a similar problem for the program itself, as it consists of an array of functions. We address this using the same approach; the beginning of every flattened bril file contains a fixed size header that stores the size of each table of contents + flat function element in the byte array that makes up the rest of the file.

## Implementation

### Zerocopy

In order to facilitate conversion between slices of bytes and our flat data structures we used the zerocopy crate. In practice this meant adding the IntoBytes and FromBytes traits to our flat representations, and specifying their byte representations using the repr attribute. It ended up being quite challenging to get this working. Zerocopy was quite finicky about what was allowable in a struct using the zerocopy traits. We ended up needing to create new “extra-flat” versions of many of our data structures to get this to work and experimenting with different repr options. For instance we discovered that zerocopy could not convert pairs of u32s, or Options to bytes, so we needed to create a new I32Pair struct that itself implemented the zerocopy traits (I32 as opposed to u32 because we used -1 to represent the case where the Option is None). We probably could have used these extra-flat data structures everywhere, rather than converting between multiple versions, but since we had written most of our flattening logic using the original data structures we decided to stick with using new extra-flat versions. 

Our final representation of a function that worked with zerocopy:

```Rust
#[repr(packed)]
#[derive(Debug, PartialEq, Clone, Immutable, IntoBytes)]
pub struct FunctionView<'a> {
    pub func_name: &'a [u8],
    pub func_args: &'a [FlatFuncArg],
    pub func_ret_ty: FlatType,
    pub var_store: &'a [u8],
    pub arg_idxes_store: &'a [I32Pair],
    pub labels_idxes_store: &'a [I32Pair],
    pub labels_store: &'a [u8],
    pub funcs_store: &'a [u8],
    pub instrs: &'a [FlatInstr],
}
```

Once we finally made zerocopy happy about all our data structures we were able to convert seamlessly between the in memory representations and bytes which we could directly mmap to disk, enabling the flat file format we envisioned as one of our goals.

When deserializing from our flat file format back to our flat data structures we ran into a very nasty bug. On some programs we would get alignment errors when attempting to convert bytes to our data structures, but we could fix/break programs just by changing the variable names! We eventually realized that a program would crash if the total bytes used to represent all variable names (or labels, or functions) was not a multiple of 4! This was because our top-level arrays just stored all names contiguously, and so would be unaligned unless the total bytes to represent all names was a multiple of 4. To fix this we just padded all of these top level arrays with null bytes.

### Interpreting flat Bril

Once we had finished implementing our infrastructure for flat Bril representations, we implemented an interpreter that operated directly on the flat data structures. We built this interpreter from the ground up rather than trying to adapt the existing Rust interpreter. This was mostly straightforward, and we were able to model our logic after the Typescript interpreter. The main challenges were introduced by needing to keep track of which array each pair of indices in our flat data structures was referring to. We had a few bugs caused by assuming that some indices referred to the top-level name arrays, when in fact we needed to go through an intermediate array. In hindsight we could have done a better job with our naming conventions to avoid this.

## Evaluation

For our evaluation, we decided to test flat bril on 70 core bril benchmarks. To check the correctness of our implementation, we used Turnt to verify that all benchmarks using our flat bril interpreter returned the same result as that of the reference Brili interpreter, for which we were successful. Additionally, to check the correctness of our infrastructure converting JSON files to/from our flattened format, we manually checked that the final json output converted back matched that of the original. To measure the performance impacts, we did the following using hyperfine:

1. Measured the CPU wall clock time (using Hyperfine) for the flat bril, brili typescript, and brili rust interpreters, comparing their performance
2. Measured the CPU wall clock time for json roundtrips (json -> flat -> json). (Although there wasn’t a specific baseline for this.)

Below is a table showing the time taken for json roundtrips. We tested this on all the core benchmarks, but due to space constraints, we only list a few here. These are averaged over 10 runs, with a warmup of 3.


| Benchmark | Mean [ms] | Min [ms] | Max [ms] | Relative |
|:---|---:|---:|---:|---:|
| `bitshift` | 451.5 ± 9.7 | 437.9 | 466.1 | 1.03 ± 0.03 |
| `call` | 453.9 ± 34.5 | 432.4 | 541.5 | 1.04 ± 0.08 |
| `const` | 521.2 ± 160.5 | 435.0 | 941.7 | 1.19 ± 0.37 |
| `euclid` | 472.9 ± 37.2 | 436.8 | 544.3 | 1.08 ± 0.09 |
| `main-args` | 442.6 ± 10.5 | 430.9 | 464.3 | 1.01 ± 0.03 |
| `montgomery` | 441.6 ± 5.7 | 434.2 | 454.4 | 1.01 ± 0.02 |
| `nop` | 521.6 ± 114.0 | 438.3 | 744.3 | 1.19 ± 0.26 |
| `perfect` | 503.1 ± 37.1 | 462.7 | 578.5 | 1.15 ± 0.09 |
| `reverse` | 486.1 ± 61.1 | 432.5 | 616.1 | 1.11 ± 0.14 |
| `rot13` | 438.5 ± 7.8 | 428.4 | 452.8 | 1.00 |


We used Hyperfine to compare the runtime of our interpreter over our flattened (mmap-ed) representation of Bril files, versus the TypeScript and Rust Brili interpreters on the JSON representation of Bril files. We ran the three interpreters on 70 Core Bril benchmarks, and for each benchmark, measured the mean execution time of the interpreter over 10 runs. From the scatter plot below, we see that Flat-Bril’s execution time is consistently in-between the TypeScript and Rust Brili interpreters (closer to the latter in many cases). 

Rust Brili outperforms our flattened interpreter for the vast majority of benchmarks, although our interpreter has a lower 

For a few benchmarks (`call`, `call-with-args`, `mccarthy91`), our flattened interpreter even has a lower mean execution time compared to Rust Brili, although 


<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="bench_results.png" alt="" style="width: 100%;">
</div>

We also used the Samply profiler to figure out where our interpreter was spending most of its time. The stack chart below (obtained from the Firefox Profiler UI) demonstrates how our interpreter exectuable spends its time on the benchmark `catalan.bril` – we chose to highlight this benchmark since it has many recursive function calls and the difference between our interpreter and the TypeScript/Rust Brili interpreters’ performance is noticeable. 

The stack chart shows that most of the runtime of the executable spent was in the interpreter-related functions: `mmap`-ing the flat file  format was relatively quick, and so was converting the JSON to a flat file format (omitted from the screenshot). Zooming into the stack chart (second screenshot below) and focusing on the function calls towards the bottom, we see that when interpreting individual individual instructions, our interpreter calls a lot of standard library functions that manipulate `HashMap`s and `Vec`s. This is because the environment datatype in our interpreter is defined as `HashMap<&str, BrilValue>` (mapping variable names to a Bril value), i.e. keys in the hashmaps are (references to) strings. We realized that strings were the canonical way to represent variables in the environment, since different indexes in the arguments field of an instruction may point to the same underlying variable. However, the fast Rust Brili interpreter canonicalises variable names into a numerical representation (as described in their blogpost), allowing for faster look-ups in their environment. We suspect that this is one of the reasons why the Rust Brili interpreter out-performs our flat interpreter. 



<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="stack_chart2.png" alt="" style="width: 100%;">
</div>


<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="stack_chart1.png" alt="" style="width: 100%;">
</div>

We also used `/usr/bin/time -l` to compare the memory usage of our interpreter to both reference Brili interpreters (the results below are also for `catalan.bril`): 

|                                   | Flat Bril | Brili (TypeScript) | Brili (Rust) |
|-----------------------------------|-----------|--------------------|--------------|
| Maximum Resident Set Size (bytes) | 18923520  | 57262080           | 19382272     |
| Page Reclaims                     | 3810      | 6401               | 3823         |
| Page Faults                       | 250       | 2036               | 203          |
| Voluntary Context Switches        | 158       | 126                | 89           |
| Involuntary Context Switches      | 210       | 2480               | 176          |

Our interpreter compares favorably to Rust Brili in terms of peak physical memory usage (max resident set size). Notably, our interpreter has an order of magnitude fewer page faults and involuntary context switches compared to the TypeScript interpreter (250 vs 2036 and 210 vs 2480), although it’s unclear whether this is due to our choice of implementation language (Rust vs TypeScript) as opposed to our data structure flattening strategy, since Rust Brili also has similar metrics. 