+++
title = "Compilation-based Execution Engine for Database"
[extra]
bio = """
 Ankush Rayabhari is a first-year MS student in Computer Science at Cornell who
 is interested in databases and compilers.
"""
[[extra.authors]]
name = "Ankush Rayabhari"
+++

## Introduction

Existing database systems initially convert a SQL query to a tree of relational algebra operators and then interpret that plan. In the past, this wasn't a bottleneck as the main cost of processing data was reading and writing data to disk. Since the amount of CPU time spent processing data was negligible compared to the disk access time, the overheads of interpretation didn't play a large role.

Nowadays, databases have access to faster storage and large amounts of main memory which means that the CPU time spent processing data is much more valuable. To reduce the overhead of interpretation, we can directly compile trees of operators down to binary and then execute them. This projects explores implementing such a compiler.

## Background

### Relational Algebra

### Interpretation

## High Level Design

## Implementation

Since the primary focus of this project is the compilation portion, I've generally omitted or simplified any database specific portions that were not necessary.

### Operator Tree



### Input

Since parsing and query planning/optimization are not the focus of this project, I've not implemented these. Rather, I manually enter a tree of relational algebra operators and then compile and execute that.

### Types

I map each of the SQL types to the following underlying type:

| SQL Type    | Implemented Type |
| ----------- | ----------- |
| SMALLINT | 16-bit signed integer |
| INT      | 32-bit signed integer |
| BIGINT   | 64-bit signed integer |
| BOOLEAN  | 8-bit boolean |
| REAL     | 64-bit floating point |
| TEXT     | Variable-length string |
| DATE     | 64-bit signed integer containing unix millisecond timestamp |

### Storage

The file format used for a table consists of each of the columns stored as a separate file. The format of each column depends on its type:
- The format for all types except for TEXT consists of tightly-packed values of that data, much like an in-memory array. No other metadata such as cardinality is stored.
- For TEXT types, the format used is as follows. The first 4 bytes represent an unsigned 32-bit integer that contains the cardinality (number of strings) inside the file. Then, for each string, we store 8 bytes. The first 4 bytes are an unsigned 32-bit integer that contains the string length. The second 4 bytes are an unsigned 32-bit integer that contains the byte offset of the string's data in the file. After this metadata array, we store all the strings in a packed format.

To avoid having to implement a buffer pool manager and on-demand paging, I mmap the column files and rely on OS paging to handle this.

### Code Generation

We generate C++ code directly to simplify the implementation. All the types above are native C++ types that we can reuse in the implementation. For example, we can use int16_t, int32_t and int64_t for 16, 32 and 64-bit signed integers respectively.

As mentioned in the design, each operator in the tree needs to have a produce/consume function on it. While initially I developed these as plain functions that simply take in an operator as input, I quickly realized that  the produce and consume function calls on the same operator needed to share state. For example, the hash join produce function outputs a hash table declaration while the hash join consume function relies on that same variable.

Because state needs to be shared, we could attach these extra state variables onto the relation algebra operators. However, for better separation of concerns, we create a wrapper translator object around each operato and then organize them into the same tree. This allows for per-operator state sharing without having to modify each of the plan classes. We use the visitor pattern to generate these translator wrappers.

Each translator needs to know which variables correspond to the schema of each operator. For example, the output operator needs to know which variables correspond to each column of the child operator's schema to know which variables to push into standard out. To handle this, each translator contains a list of variables that can be queried, one for each column of its output schema.

### Code Execution

Once the C++ file has been generated, we invoke clang on the file, allowing any C++ header defined in the codebase to be included. This enables us to reuse classes defined in the codebase such as the ones for file reading. This generates a dynamic library that we load into the database with dlopen. We retrieve the aforementioned compute function via dlsym and then execute it.

## Performance Evaluation

## Personal Experiences

The largest barrier I had to overcome in working on this project was understanding all the pieces of the puzzle. I needed to understand each portion of the system enough to implement a reduced version of it.

## Future Work

## Sources

I used the following sources as references: