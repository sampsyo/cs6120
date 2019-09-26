+++
title = "C Implementation of Bril"
extra.author = "Bhargava S. Manja"
extra.bio = """
  [Bhargava S. Manja](https://github.com/bhargee) is a first year PhD student
interested in computer vision, fast computing, and domain specific languages
"""
+++

## Project Report 1: C Implementation of Bril

While bril is a very useful testbed for exploring existing language
technologies and experimenting with new ideas, I was frustrated with the
tooling around the language. It took me three or four hours of fiddling around
with nodejs, npm, and python to get `brili` working, and I did not get either
`bril2json` or `bril2txt` to work on my machine (I reimplemented them myself in
python). I've rarely had such issues with my trusted systems programming
language, C. I decided to implement a simple, fast, and correct interpreter for
bril. I call it `cril`.

All code can be found in the [project
repository](https://github.com/Bhargee/cril)
  
## Design 

The goal here is simplicity and speed (in comparison to other student
interpreters). The only external library used was for [parsing
JSON](https://github.com/kgabis/parson). The interpreter's evaluation loop, in
`src/interp.c`, first loops through the parsed bril program and notes the
indices of labels. This was the simplest way to implement jumps and branches,
which become a simple setting of the instruction pointer to the label's index
(or an error if the label is not found in the label->index map). Actual
instructions are implemented with a set of functions, one for each op code.
Once the `op` is known, teh interpreter calls one of these functions., which
fetches arguments, does the required manipulation, and stores its result in the
right place (or in the case of effect operations, has teh correct effect). 

## Op Code Implementation
All instructions but `jmp`, `br`, `print`, and `const` have a trivial
implementation. A function called `get_or_quit` fetches arguments from the
source program and stores their values in a global array, sized to the max of
the argument count of all non-print instructions (2). If the variable is not
found in storage, an error is reported (containing the specific issue,
incorrect variable, and instruction pointer value) and the program quits.
Otherwise, the op implementing function calls `put` with the argument that
implements the specific op code, which stores the result in the destination
variable.. For example, the `add` op code is implemented
as follows:

```c
static void op_add() {
  get_or_quit();
  put(mem_args[0]+mem_args[1]);
}
```
Above, `mem_args` is the name of the global argument array. Thus, most op codes
have 2 line implementations. Even `br`, the most complex op code, requires only
16 lines to implement. 

## Memory Implentation (and More, for Free)
I needed a hash table implementation to implement variable storage, so I built
it myself. It can be found in `src/table.{c,h}`. The dictionary uses open
addressing and linear chaining. The hash function copies the hash used by
java.lang.String’s hashCode() method. The `_hash` function computes
a polynomial whos coeffecients are the integer values of the string's
characters, evaluated at `x=31`. The polynomial is evaluated with Horner's
method. The `table` maps string keys to `int64_t` values.  I represent bril
integers and bril booleans with `int64_t` to avoid storing the type of bril
variables. Since bril's typesystem only allows for those two types, this design
is sufficient for now. Incorporating more types will be simple: I can extend
the `table_elem` struct with a type bitfield/enum value.  For now, this simple
table suffices. 

The table code is also used for the label->index map and to store the mapping
between string op codes and the index for that op code's implementing function
in an array of function pointers.

## Evaluation
### Correctness
I ran `cril` against the programs in `bril/test` and made sure the outputs
matched `<program>.out`. My interpreter gave correct results on a fibonacci
program posted in the class slack channel, on which the reference interpreter
gave incorrect results. 

### Performance
I wanted to have a way to get reliable performance numbers to square off
againts any other student's implementation of bril. First, I had the main
evaluation loop return a `uint64_t` number of nanoseconds of elapsed time.
I used the POSIX provided `timespec` structure to record the time tracked by
`CLOCK_PROCESS_CPUTIME_ID`, the nanosecond resolution process time clock. This
tracks CPU ticks spent on the program process itself, irrespective of other
scheduled processes. The code is reproduced below:
```c
  JSON_Array *instrs = get_main_func_instrs(fn);
  struct timespec tick, tock;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tick);
  mem = table_init();
  labels = table_init();
  make_dispatch();

  num_instrs = json_array_get_count(instrs);
  
  for (size_t i = 0; i < num_instrs; ++i) {
    curr_instr = json_array_get_object(instrs, i);
    const char *label = json_object_get_string(curr_instr, "label");
    if (label) {
      table_put(labels, label, i);
    }
  }

  size_t op_ind = -1;
  char const *op = 0;
  bool not_found = false;
  for (ip = 0; ip < num_instrs; ++ip) {
    curr_instr = json_array_get_object(instrs, ip);
    op = jogs(curr_instr, "op");
    if (!op) continue; // label
    op_ind = (size_t) table_get(disp, op, &not_found);
    if (not_found) {
      quit("Operation not defined", op);
    } else {
      (op_funcs[op_ind]());
    }
  }

  cleanup();
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tock);
  return (1000000000 * (tock.tv_sec - tick.tv_sec) + tock.tv_nsec - tick.tv_nsec);
```

Notice the following:
* I start timing on line 2, after JSON parsing. I did not want that to be
  included
* I stop timing AFTER cleaning up all data structures used for interpretation
* The last line returns the time elapsed in nanoseconds

Then, in `src/main.c`, I have a constant named `NUM_RUNS`, and if `cril` is
called with `--benchmark`, it will run each program under `tests` `NUM_RUNS`
times, calculate a mean and standard deviation per program, and output the
results. I've included some sample output for `NUM_RUNS=10000"

```
NUM RUNS: 10000
test/interp/
--------------------------------------
test/interp/jmp.json
MEAN: 0.004536
STDV: 0.003142

test/interp/div.json
MEAN: 0.003799
STDV: 0.001767

test/interp/ret.json
MEAN: 0.003404
STDV: 0.001907

test/interp/nop.json
MEAN: 0.003787
STDV: 0.002202

test/interp/br.json
MEAN: 0.004586
STDV: 0.002476

test/interp/fib.json
MEAN: 0.108300
STDV: 0.029558

test/parse/
--------------------------------------
test/parse/print.json
MEAN: 0.003200
STDV: 0.002105

test/parse/comment.json
MEAN: 0.002641
STDV: 0.001431

test/parse/add.json
MEAN: 0.003943
STDV: 0.002204

test/print/
--------------------------------------
test/print/add.json
MEAN: 0.004013
STDV: 0.002048
```
