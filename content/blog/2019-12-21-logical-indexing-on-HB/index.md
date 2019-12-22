+++
title = "Logical Tensor Indexing for SPMD Programming"
[extra]
katex_enable = true
latex = true
[[extra.authors]]
name = "Edwin Peguero"
+++

My current research explores programming abstractions for an experimental, parallel architecture (codenamed HammerBlade).
The language feature prototyped in this project aims to simplify *SPMD* programming, wherein all nodes apply the same kernel over different pieces of input data.

The need to distribute inputs over cores in SPMD programs introduces problems that are orthogonal to the kernel, but that nonetheless affect implementation:
- Data format of inputs/outputs: 
  Each tensor $A$ must be encoded in a particular **format**, $f$, that maps the **logical index** of each tensor element $A<x><y>$ to **physical index** $f(x,y)$, s.t., $A<x><y> = A'[f(x,y)]$.
  Low-level languages like C, the target language in the current implementation, require this dimensionality reduction to maximize performance contributors such as data locality and bulk memory allocations/movements.
  Different formatting choices lead to *different implementations* of the *same kernel*.
  We constrain our prototype to use the **row-major order** format, which sequentially lays out rows of $A$ via $f_A(x,y) = x * stride + y$, where $stride$ indicates the number of elements per row of $A'$.
  Our results generalize to similar formats (e.g., *col-major order*), and we leave exploration of more exotic formats to future work.

- Caching:
  Preemptive promotion of required inputs in the memory hierarchy, another form of data locality, often leads to significant performance gains.
  However, the choice of caching can change data formatting. 
  For instance, if a **row-major order** sub-matrix is cached from a **row-major order** matrix, its stride will be different.
  Thus, the choice of caching also leads to *different implementations* of the *same kernel*.

This project implements a formal separation of **logical indexing** from **data formatting** via a simple 2-D stencil algorithm example.

# Stencil Algorithm Using Logical Indexing
The proposed syntax for logical indexing uses angle brackets `A<x><y>` to distinguish from traditional array indexing `A[x][y]`.
We demonstrate this language feature by means of a 2-D stencil algorithm that computes 2-D tensor $B<x><y>$ from tensor $A$ as the average of $A<x><y>$ and its neighbors. 
At the boundaries of $B$ (i.e., $B<0><y>$, $B<n-1><y>$, $B<x><0>$, $B<x><m-1>$), the algorithm simply assigns the input directly.
In SPMD style, each core transforms a block of data, denoted $A_tile$ into a block of output, denoted $B_tile$.
The pseudo-code below uses logical indexing to capture this behavior; noting that a more efficient implementation is possible but unnecessary for purposes:
``
for (x : B_tile.row_ix in B_tile.row_range) {
  for(y : B_tile.col_ix in B_tile.col_range)  {
    if(0 < x as B.row_ix < B.max_row-1 &&
       0 < y as B.col_ix < B.max_col-1) {
      B<x><y> = neighbor_avg(A<x><y>);
    }
    else {
      B<x><y> = A<x><y>;
    }
  }
}
``
Both loops iterate over zero-based logical row and column ranges, `B_tile.row_range` and `B_tile.col_range`, with a branch detecting points at a logical boundary where behavior differs from inner points.

To determine that an index from sub-tensor $B_tile$ resides on the logical boundary of parent tensor $B$, we perform **index casting** from `B_tile.row_ix` to `B.row_ix` and analogously for columns.


# Specifying the Data Format
By expressing the stencil algorithm in terms of logical indexing, we disentangle it from implementation details of data formatting and caching.
We can specify these details by decorating inputs and outputs in the kernel signature as follows:
``stencil_kernel(input[blocked cached, row-major] A : float[n][m], output[non-cached, row-major] B: float[n][m])``
This signature specifies that both `A` and `B` are formatted in row-major order, that a block of `A` will be cached by each core, and that cores will write directly to `B` in global memory.
Such a specification suggests to the compiler particular implementations for data formats $f_A, f_B$.

# Stencil Algorithm with Logical Indexing in C
Logical indexing and data formatting will be incorporated into my research language; however, compilation for this syntax has not been implemented yet. 
Here, I share a prototype of logical indexing in C that generates cached and non-cached variants of the above stencil algorithm.


We'll take the stencil algorithm as a diving board into the interplay of logical indexing and data formatting:
```c
// stencil algorithm
void stencil(tensor A, tensor B) {
  for (int x = 0; x < B.n; x++) {
    for (int y = 0; y < B.m; y++) {
      if (0 < log_to_phys_row(B, x) &&
            log_to_phys_row(B, x) < top_level_tensor(B)-1 &&
          0 < log_to_phys_col(B, y) &&
            log_to_phys_col(B, y) < top_level_tensor(B)-1) {
        log_write(B, x, y, avg(A, x, y));
      }
      else {
        log_write(B, x, y, log_read(A, x, y));
      }
    }
  }
}
```

The translation from the earlier psuedo code is nearly straightforward: logical reads and writes are respectively denoted `log_read` and `log_write`.
Note the `tensor` type, which collects data formatting information of an array to implement logical data operations; it's pretty [dope](https://en.wikipedia.org/wiki/Dope_vector):

```c
typedef struct tensor
  { struct tensor *parent;
    float* arr;
    int n; int m;
    int stride; 
    // physical offset of logical origin in parent
    int phys_row_off; int phys_col_off; 
    bool cached;
  } tensor;
```

Using this information, we implement `log_read` as follows:
```c
  float log_read(tensor *A, int log_row, int log_col) {
    int phys_row = log_to_phys_row(A, log_row);
    int phys_col = log_to_phys_col(A, log_col);

    return A->arr[phys_row * A->stride + phys_col];
  }
```

`log_to_phys_row` applies sub-tensor offsets to the logical row index recursively up the nested sub-tensor structure:
```c
int log_to_phys_row(tensor *A, int log_row) {
  return A->cached || A->parent == NULL ? 
    log_row + A->phys_row_off:
    log_to_phys_row(A->parent, log_row + A->phys_row_off);
}
```

Importantly, we do not recurse over cached sub-tensors to compute the physical index, since such tensors are only logically, but not physically, nested inside the parent tensor.

We calculate logical upper bounds by recursively traversing up the `parent`s of sub-tensors and returning the max row or column of the top-level `parent`.
Caching plays no role in calculating this, or any, logical quantity.

```c
tensor* top_level_tensor(tensor *A) {
  return A->parent == NULL ? A : top_level_tensor(A->parent);
}
```

Unfortunately, index casting is not actually implemented yet.
Instead, the above implementation cheats by using the `log_to_phys` functions: rather than perform logical boundary-checking it performs physical boundary checking.
This works for our purposes, since the logical and physical boundaries are equivalent for the row-major order data format.

# Cached vs Non-cached Implementations

Both the cached and non-cached implementations define tensors as follows:
```c
  int tile_n = n / cores_x;
  int tile_m = m / cores_y;

  int b_base_row = tid_x * tile_n;
  int b_base_col = tid_y * tile_m;

  tensor a =
    { .arr = a_arr,
      .n = n, .m = m,
      .stride = m };
  tensor b = 
    { .arr = b_arr,
      .n = n, .m = m,
      .stride = m };
  tensor a_tile = 
    { .arr = a_arr, 
      .parent = &a,
      .n = tile_n, .m = tile_m,
      .stride = m,
      .phys_row_off = b_base_row, .phys_col_off = b_base_col};
  tensor b_tile = 
    { .arr = b_arr, 
      .parent = &b,
      .n = tile_n, .m = tile_m,
      .stride = m,
      .phys_row_off = b_base_row, .phys_col_off = b_base_col};
```

`a_tile` and `b_tile` represent the sub-tensors from input `A` and output `B` that are operated on by the current tile.

The **global memory (non-cached) implementation** uses these tensor definitions to run the stencil algorithm as follows:

```c
  stencil(&a_tile, &b_tile);
```

The **cached implementation** defines the same tensors, but additionally performs a caching routine that moves data from tensor `a_tile` to a new tensor `a_scr`, as follows:
```c
...
tensor a_scr =
  { .arr = scr,
    .parent = &a_tile,
    .n = tile_n+2, .m = tile_m+2,
    .stride = m+2,
    .phys_row_off = 1, .phys_col_off = 1,
    .cached = true};

  cache(&a_tile, &a_scr);
  stencil(&a_scr, &b_tile);
```

The caching strategy reads a window around the tile data consisting of two additional rows and columns:
```c
void cache(tensor *input, tensor *cache) {
  for (int x = -1; x < input->n+1; x++) {
    for (int y = -1; y < input->m+1; y++) {
      if(0 < log_to_phys_row(input, x) &&
          log_to_phys_row(input, x) < top_level_tensor(input)->n &&
         0 < log_to_phys_col(input, y) &&
          log_to_phys_col(input, y) < top_level_tensor(input)->m ) {
        log_write(cache, x, y,
            log_read(input, x, y));
      }
    }
  }
```

This caching routine shows how we can use the physical offset to arbitrarily shift the otherwise zero-based logical indexing.
Using physical offset for this purpose is also cheating, since logical indexing may in principle be changed while preserving physical indexing.


# Evaluation: Cached vs Non-cached
Both stencil implementations ran on a gem5 simulation of a 4x4 HammerBlade architecture.
Both stencil implementations were tested using matrices with elements equal to their row-major order indices. 
This choice of matrix allowed for simple confirmation of correctness, since these matrices are not transformed by the stencil algorithm.

We compare performance of both implementations by reading the `numCycles` statistic produced by the simulation for each input size and averaging the result over the four cores, and over three trials:

**non-cached**
| 4x4     | 8x8     | 16x16   |   34x34 |
|---------|---------|---------|---------|
|  7605   | 28740   |  123769 |  580502 |
|         |         |         |         |

**cached**
| 4x4     | 8x8     | 16x16   | 34 x 34 |
|---------|---------|---------|---------|
|  14592  | 35961   |  115200 | 474928  |
|         |         |         |         |

# Discussion
These results demonstrate that the cost of caching is amortized with the size of the input.

Although this result is expected, I wondered why the cost of caching seemed so high.
Cursory testing revealed that removing the branching from the cache loop removed ~10k cycles from the 4x4 and 8x8, ~40k cycles from the 16x16 input, and ~130k cycles from 34x34.
These results suggest that a more efficient caching routine that minimizes branching would improve performance.

Further analysis can be done to determine how much the simulation penalizes global memory vs scratchpad operations.
This analysis would also inform whether caching on the output might present an opportunity for additional performance benefits.


# Conclusion

Thanks to logical indexing, we generated two implementation from one high-level stencil algorithm.
I plan to use the flexibility and expressiveness afforded by logical indexing as a foundation to implement future language features.
