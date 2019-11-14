# Strength Reduction for Multiplication

## Motivation

The strength reduction method is to replace expensive operations with cheaper but equivlent one, so we can obtain faster program. For this project, we will focus on the weak form of strenth reduction. Specifically we focus on how to replace constant multiplications with cheaper operations.

Most modern processors have different latencies and throughputs for different kind of instructions. It is somtimes possible to find instructions which are mathematically equivelent but fasters in practice. For example, on some processors, mutiplication may run slower than bitwise shift. Therefore, it is possible to replace multiplication `x * 2` with bitwise left shift operation `x << 1` for a better performance.

## Alternations for Multiplication with Constants

Strength reduction for multiplication with a constant of powers of $2$ is obvious. However, even the constant is not a power of $2$, reducing multiplications to bitwise shifts is still possible.

Since constants can be represented as sum of powers of $2$, we can use sum of bitwise shifts to replace multiplication operations. For example, `x * 7` can be represented as `(x << 2) + (x << 1) + x`.

$7$ can be also represented as $8-1$, a power of $2$ minuses a constant. In fact, $7$ is "closer" to the next number. If we reduce `x * 7` to `x * (8 - 1) = x * 8 - x * 1 = (x << 3) - x`, it requires less number bitwise shifts and add/minus operations.

Therefore, we have three choices for mulpliying a constant:

1. multiply the constant directly
2. binary decompose the constant, and sum up the results of bitwise shifts
3. represent the constant as $2^k-c$, left shift $x$ by $k$ bits, and binary decompose $c$ then substracts those bitwise shifts

In order to determine with methods we want to use for a mulplication reduction, we need a cost function to compare the cost of those instructions. 

Based on different architecture, we can assign a cost to each of 

1. bitwise shift operation
2. add/minus operation
3. multiplication

The function calculates the total costs of these three approaches and determines which one has the lowest cost.

## Evaluation

### Evaluation Method

In order to show how much cost it can reduce by replacing multiplications with bitwise operations, we implement a llvm pass to insert two instruction counters: one is to count the cost of the original instruction and the other is to count the cost of optimized instruction. For each mulitplication instruction, we instrument instructions to update the costs.

## Our Thoughts

On most of modern processors, the performance difference between add/minus/shift operations and multiplication operations are not that huge. In a non-scientific computation workload, the optimization strength reduction for multiplication is negligible.

We also found that the strength reduction could be very useful on hardware design languages, such as Verilog for FPGA design. Since multiplication circuit has longer path than add/minus/shift operations, strength reduction may be used to reduce the length of critical path and total number of gates of a circuit.
