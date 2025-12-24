+++
title = "Using Constraint Programming for tiling and fusion in tensor programs"
[extra]
bio = """
  Amanda Wang and Nathaniel Young are both CS PhD students at Cornell.
"""
[[extra.authors]]
name = "Nathaniel Young"
link = "https://natetyoung.github.io"  # Links are optional.
[[extra.authors]]
name = "Amanda Wang"
+++

We wanted to build a system to automatically find optimal tiling and fusion strategies for short tensor programs using powerful solvers for NP-Hard problems. We focused on the "chains-of-einsums" setting studied by [this "Orojenesis" paper](https://people.csail.mit.edu/emer/media/papers/2024.06.isca.orojenesis.pdf) from ISCA 2024: a list of tensor operations, each with one input taken from the previous one's output. For simplicity, we restrict ourselves to einsums where the tensors are indexed by simple subsets of the iteration variables, rather than arbitrary affine functions of them. This allows us to represent things like matmul, batched matmul, elementwise operations, etc., but unfortunately not things like convolution.

Our overall goal is easily understood by comparison: both this project and Orojenesis consider the problem of finding tiling and fusion strategies for minimizing the number of accesses made beyond some cache level, given the size of the cache. However, while Orojenesis finds optimal strategies using exhaustive search over tilings and specific constraints for fusibility which depend on einsum type, we wanted to come up with a solver-based approach which encodes fusibility in a general way for a wide range of einsums. Also, while Orojenesis finds a Pareto-optimal set of design points in terms of cache size and accesses, we designed programs to find the strategy with the fewest accesses given a fixed cache size, although essentially the same programs could be used to construct an Orojenesis-style Pareto frontier in a solver which supports that.

# Approach

We designed a series of constraint programs to solve tiling and fusion problems using [CP-SAT](https://developers.google.com/optimization/cp/cp_solver).

In order to see how we designed these programs, let’s consider how to think about loop tiling in the first place.

### How to think about tiled loops

Here's some pseudocode for an untiled matrix multiplication:

    for m in [0, 1024):
        for n in [0, 1024):
            for k in [0, 1024):
                C[m, n] += A[m, k] * B[k, n]

And here's a tiled one:

    for m0 in [0, 1024/128):
        for n0 in [0, 1024/128):
            # --- Blocking Level for C: inside here, C remains in cache ---
            for k in [0, 1024):
                # --- Blocking Level for B: inside here, B remains in cache ---
                for m1 in [0, 128):
                    # --- Blocking Level for A: inside here, A remains in cache ---
                    for n1 in [0, 128):
                        C[m0*128+m1, n0*128+n1] += A[m0*128+m1, k] * B[k, n0*128+n1]

The code above has blocking levels indicated with comments. If we were on a device with explicit data movement, we’d replace them with data movement commands, like so:

    for m0 in [0, 1024/128):
        for n0 in [0, 1024/128):
            C_local = zeros(128,128)
            for k in [0, 1024):
                load B[k:k+1, n0*128:(n0+1)*128-1] as B_local
                for m1 in [0, 128):
                    load A[m0*128+m1:m0*128+m1+1, k:k+1] as A_local
                    for n1 in [0, 128):
                        C_local[m1, n1] += A_local[0, 0] * B_local[0, n1]
            store C_local in C[m0*128:(m0+1)*128-1, n0*128:(n0+1)*128-1]

This suggests a way to check whether the tiled loop nest is valid and estimate its performance.
For performance, given the intended blocking levels, count the number of times each buffer needs to be loaded or stored as the product of all loop bounds outside its blocking level, and the size of the buffer in the cache as the product of the "participating" loop bounds inside the level (that is, loop bounds whose iterate changes which element is accessed in this tensor. In matmul, for instance, the size of a loop over k does not change the blocked size of the C matrix). The product of these numbers is the total number of accesses done for the tensor; we want to minimize the sum of accesses.

For validity, ensure that for each dimension in the original einsum, the product of its factors in the tiled loop nest is at least its total size. To ensure we don't overflow the cache, the sum of buffer sizes in the cache should be at most a given capacity.

So how can we find a tiled loop nest of optimal performance? As a first attempt at expressing the optimal tiling problem using CP-SAT, let’s try splitting each dimension into a fixed number of tiled loops (say 2) and then let the solver choose:
1. The order of the factored loops
2. The factor size in each loop
3. The blocking level of each tensor
This works fine; we represent the factor sizes with integer variables, and the positions of the loops and blocking levels using boolean variables. For a 6-loop schedule of matmul, for instance, we will have 36 boolean variables of the form "loop __i__ is assigned to position __j__" and 18 boolean variables of the form "position __j__ is inside the blocking level for tensor A," plus 6 integer variables for factor sizes. We include constraints for validity as described above and ensure a one-to-one mapping between loops and positions. For simplicity, we also enforce that there not be any partial spilling of the output tensor (meaning that the "reduction" dimensions which do not participate in the output tensor cannot be outside its blocking level).

Note that this has some redundancy, though. We’re asking the solver to choose the exact position of every loop, but we only care whether each loop is inside or outside each blocking level. To get rid of this, we can try asking the solver to instead choose just whether each loop is inside or outside each blocking level, with the constraint that an order on loops can somehow be formed. We encode that by asking the solver also to choose a partial order over loops consistent with the blocking levels: variables of the form __i < j__, which must be true whenever __i__ is inside some blocking level and __j__ is outside it, and which must obey transitivity and noncontradiction constraints. A consistent total order over loops can be extracted later. This formulation is available [here](https://github.com/natetyoung/tensor-program-dse/blob/main/po_einsum_cpsat97.py).

This is still a little redundant, though. We’re not making the solver choose a total order, but it's still _allowed_ to choose one, so it might be spending extra time proving to itself that the exact order doesn't matter. Another issue is that we have a few poorly-justified assumptions in our model. Why 2 factors of each dim specifically? Do we need more? Can we get away with fewer in some cases (for instance, perhaps for reduction dimensions, we may not have to tile them if we are already enforcing that there is no partial spilling)?

We can take our approach further in order to solve both of these problems at once. If all that matters is whether each factor is inside or outside each blocking level, let's not create dimensions and choose their order; let's choose _the order of the blocking levels_ first and then choose how much of each dim goes into each "bucket" created between them. In the tiled matmul example above, for instance, we have 4 buckets, formed by the 3 blocking levels (in order C, B, A). The outermost contains factors `M/128` and `N/128` of `M` and `N` respectively; the second bucket has all of dim `K`; the third has factor `128` of `M`, and the innermost bucket has factor `128` of `N`. To encode these choices in a CP-SAT program, we create 9 boolean variables of the form "the blocking level for tensor __X__ is assigned to position __j__", plus 12 integer variables for "the factor of dimension __i__ used in bucket __j__." Then, we can use auxiliary variables to encode whether each bucket (and therefore the factors in it) is inside or outside each blocking level, and so we can again encode validity and performance as above. This formulation is available [here](https://github.com/natetyoung/tensor-program-dse/blob/main/cb_einsum_cpsat97.py).

Something might strike us about the factors in these buckets, though. Consider what would happen if we put any factor of `M` in the innermost bucket above. This dimension does participate in A, so it is increasing the size of A in the cache. If we took this factor and moved it one bucket outwards, the size of A would decrease by that factor, and the number of accesses done for A would not change at all; neither would any cost for any other buffer, since the factor has not moved across their blocking level. So, any factor greater than 1 in a bucket where that factor's dimension participates in the tensor whose blocking level forms the outer boundary of the bucket __cannot be part of a Pareto-optimal solution__, and so we can constrain all such factors to 1. A similar argument can be made for, say, a factor of `N` in the second-to-innermost bucket of the matmul above; the factor could be moved to the innermost bucket in order to decrease the total number of accesses for A without changing anything else. Adding these constraints is why we call this model "_constrained_ bucketing," and it is a major part of why the solver is able to solve it for even large matmuls in well under a second (the formulation linked here calls this optional constraint `enforce_optimal_placement`).

### How to think about fusion

A major insight to note about the way of thinking about tiled loops above is that the blocking level of a buffer is "the position in the loop where data movement for the buffer is sequenced with the compute". With this intuition -- that a blocking level is simply where an einsum "gets the data" for that buffer, we can think about fusion: two einsums can be fused, with one passing its output tensor to another, if their factors outside that tensor's blocking level are identical. The blocking level becomes the "fusion level" for that tensor.

We leveraged this to build a formulation for fusing multiple einsums [here](https://github.com/natetyoung/tensor-program-dse/blob/main/cb_multifused_cpsat97.py). This formulation chooses the tiling for every einsum in the chain simultaneously, subject to the constraint that tensors passed through fusion always take the outermost blocking level, so they all share exactly one bucket and can be fused like so:

    for m0 in [0, M0): # shared factors which participate in all passed tensors
        for n0 in [0, N0):
            ...
                C = einsum0(A, B)
                E = einsum1(C, D)
                H = einsum2(E, F, G)
                ...

However, this is not optimal. Sometimes the einsums should be fused on different levels; sometimes tensors other than the passed tensor should take the outermost blocking level. Unfortunately, there does not seem to be any simple way to modify the above program to fix this. We had to go with a totally new approach.

This "full-tree" approach is [here](https://github.com/natetyoung/tensor-program-dse/blob/main/full_tree_cpsat97.py). Rather than building orders of blocking levels in multiple einsums individually, each creating their own buckets, this CP-SAT formulation creates a tree structure among all the tensors in the entire program. Instead of constraints on "factors outside this blocking level," this program has constraints on "factors which are ancestors of this blocking level in the tree," and so on. In effect, this CP-SAT program synthesizes a restricted form of a [schedule tree](https://acohen.gitlabpages.inria.fr/impact/impact2014/papers/impact2014-verdoolaege.pdf) to create an optimal tiling and fusion for an input chain of einsums.

For a 2-GEMM example given in the Orojenesis paper (`32k x 4k x 16k` and `32k x 16k x 4k`), it takes 0.76 seconds on my laptop.

# The hardest part

Constraint programs in CP-SAT are fairly difficult to debug, since if the program turns out to be infeasible there's very little way of figuring out why, and if an expected output isn't found, it can be difficult to figure out exactly which constraints it violated. For this reason, every CP-SAT formulation took some time to get right, but by far the hardest was the final "full-tree" program, especially the part of the program to estimate which buffers need to coexist with which other buffers, so we could get a total necessary cache size. Also, we found a bug in recent versions of CP-SAT and had to revert to an earlier version.

# Evaluation

We developed a flow to emit C code from the output of the full-tree solver for tiling and fusing some real-world tensor programs, and verified that they improved performance relative to a naive baseline. We then compiled both the optimized and naive implementations using clang with -O3 (at first, without vectorization or loop unrolling).
We used this flow to evaluate the solver on some real-world tensor operations. We follow the evaluation criteria used in Orojenesis by generating some C code with dummy values, which models a chain of 3 einsums (Q_proj, QK, QKV) from an inference pass through a GPT-style block. Notice that the output of each einsum is fed as input to the next, allowing for loop fusion. To vary the sizes of these operations, we adjusted the corresponding "sequence length" and "model dimensions" used to generate the einsums in the test code, and used a spread of different sizes: (`seq`, `dim`) = (`32, 128`), (`64`, `256`), (`128`, `512`), (`1024`, `4096`). We refer to them as `tiny`, `small`, `med`, and `large`, respectively. Note that the dimensions under `large` actually correspond to realistic dimensions used in the [Cerebras GPT-3-6.7B](https://huggingface.co/cerebras/Cerebras-GPT-6.7B) block.

More concretely, we used the "full-tree" approach to decide the optimal tiling and fusion for the input chain, and then generated C code with accordingly tiled loops. For each input size, we ran our optimization with capacities `4*1024`, `8*1024`, `16*1024` (in number of floats) to generate three separate optimized programs. To gauge how well our optimizer fared against code compiled with standard optimizations, we timed our solver-optimized test code and naive test code, both compiled with -O3 with vectorization and loop unrolling disabled. The table below gives the time elapsed for each test size + optimization option, averaged over 5 runs.

size | option | capacity | avg_time_sec
-----|--------|----------|-------------
tiny |unopt |N/A|0.0008058
tiny |opt |4096|0.0003706
tiny |opt |8192|0.00047799999999999996
tiny |opt |16384|0.0003878
small|unopt |N/A|0.0067888
small|opt |4096|0.004105
small|opt |8192|0.0030004
small|opt |16384|0.0029158
med  |unopt |N/A|0.0358932
med  |opt |4096|0.028004
med  |opt |8192|0.025999
med  |opt |16384|0.028292400000000002
large |unopt |N/A|62.1979504
large |opt |4096|11.396286800000002
large |opt |8192|17.026373
large |opt |16384|92.80004540000002


Our optimized code compiled with -O3 consistently outperformed the naive code compiled with -O3, which we consider a success. The exception is the large size tiled to capacity `16*1024`, which performed worse than the unoptimized code. We're not sure why this is, but we suspect that this could be approaching the L1 cache size of the architecture we were running on, which for some cores is 64k, corresponding to 16k floats. One thing to note is that while our optimized C program does already tile the loops to fuseable shapes, it doesn't directly fuse them. We also experimented with manually fusing the code, but found that the times didn't change, suggesting that -O3 was probably fusing them automatically, as expected. We later explicitly verified that this was the case by comparing the number of loops in the IR generated by compiling with -O3 and from compiling with -O0. For instance, `tinygpt_opt_4096.ll` had only 12 loops, while `tinygpt_opt_4096.O0.ll` had 25 (7 were from array memory allocations that were transformed into bulk initializations).

We later also checked whether our optimization improved vectorization by turning vectorization and loop unrolling back on for all runs. In general, we do see slight improvements for larger sizes when compared to the unoptimized code. However, at smaller sizes, the tiled code was slower than untiled when compiled with vectorization. Note that these runs were completed on an M1 instead of M4 architecture, so the numbers in this table are not directly comparable with those in the earlier table; for example, we see that the large size tiled to both capacities `8*1024` and `16*1024` see worse performance than capacity `4*1024`. 

size | option | capacity | avg_time_sec
-----|--------|----------|-------------
tiny|unopt|N/A|0.0005678
tiny|opt|4096|0.0005574
tiny|opt|8192|0.0007172
tiny|opt|16384|0.0006257999999999999
small|unopt|N/A|0.006808600000000001
small|opt|4096|0.0051827999999999996
small|opt|8192|0.0040224
small|opt|16384|0.0063196
med|unopt|N/A|0.055288000000000004
med|opt|4096|0.032602599999999995
med|opt|8192|0.030772
med|opt|16384|0.0307956
huge|unopt|N/A|82.5121808
huge|opt|4096|15.1937498
huge|opt|8192|70.1193976
huge|opt|16384|73.689952

We initially also wanted to evaluate against Polly, since it seemed like the more appropriate, "expert" comparison for loop tiling and fusion. The plan was to also benchmark against naive code that was compiled with -polly, which is LLVM's polyhedral optimizer for data-locality and loop optimizations. However, getting Polly to run on our generated code turned out to be nontrivial, and we eventually put it aside.

# Use of AI

We used Github Copilot fairly extensively in creating the CP-SAT programs; we would write the variables and constraints in plain English in a comment at the beginning of the function, and then try to get Copilot to create them in actual code later on. It worked well enough to be useful, but Copilot did make plenty of mistakes. In one instance, when the comment gave a constraint with the form "A implies B and C", it would only ever create constraints which effectively said "A and B imply C"; no amount of clarification with extra comments seemed to fix this.

We also used AI fairly extensively for generating test C code from einsums and from the optimizer. It was useful for the former, and much less useful for the latter. Nudging the LLM to produce code that properly tiled the loops turned out to be way more of a time sink than restarting and only letting it produce small modular bits of code.
