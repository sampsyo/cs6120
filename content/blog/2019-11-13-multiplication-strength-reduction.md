# Strength Reduction for Multiplication

## Motivation

The strength reduction method is to replace expensive operations with a cheaper but equivalent one, so we can obtain a faster program. For this project, we will focus on the weak form of strength reduction. Specifically, we focus on how to replace constant multiplications with cheaper operations.

Most modern processors have different latencies and throughputs for different kinds of instructions. It is sometimes possible to find instructions that are mathematically equivalent but fasters in practice. For example, on some processors, multiplication may run slower than a bitwise shift. Therefore, it is possible to replace multiplication `x * 2` with bitwise left shift operation `x << 1` for better performance.

## Alternatives for Multiplication with Constants

The strength reduction for multiplication with a constant of powers of $2$ is obvious. However, even the constant is not a power of $2$, reducing multiplications to bitwise shifts is still possible.

Since constants can be represented as sum of powers of $2$, we can use sum of bitwise shifts to replace multiplication operations. For example, `x * 7` can be represented as `(x << 2) + (x << 1) + x`.

$7$ can be also represented as $8-1$, a power of $2$ substracts a constant. In fact, $7$ is "closer" to the next number. If we reduce `x * 7` to `x * (8 - 1) = x * 8 - x * 1 = (x << 3) - x`, it requires less number bitwise shifts and add/substract operations.

Therefore, we have three choices for multiplying a constant:

1. multiply the constant directly
2. binary decompose the constant, and sum up the results of bitwise shifts
3. represent the constant as $2^k-c$, left shift $x$ by $k$ bits, and binary decompose $c$ then substracts those bitwise shifts

In order to determine with methods we want to use for a multiplication reduction, we need a cost function to compare the cost of those instructions. 

Based on a different architecture, we can assign a cost to each of 

1. bitwise shift operation
2. add/subtract operation
3. multiplication

The function calculates the total costs of these three approaches and determines which one has the lowest cost.

## Evaluation

In order to get an estimation of how much cost can be saved by multiplication strength reduction, we have run the cost analysis that reports the average cost reduced of multiplying each integer constant less than a given upper bound $n$.

In this analysis, we scaled the cost of add/subtract operations and bitwise operations to $1$ unit. According to [Agner's benchmarks on different AMD processors](https://www.agner.org/optimize/instruction_tables.pdf), multiplication operations could cost $2-16$ times more clock cycles compared to add/subtract operations. In the following table, we show the expected proportion of clock cycles it can save by multiplication strength reduction on the different cost of multiplication operations and range of the constant factor.

||n<=128|n<=1024|n<=8192|
|---|---|---|---|
|cost(MUL)=2|0.034884|0.005854|0.000915|
|cost(MUL)=4|0.143411|0.032683|0.006469|
|cost(MUL)=8|0.478682|0.207073|0.065605|
|cost(MUL)=16|0.739341|0.588293|0.430108|

From the table, we can find that the larger the performance gap between `ADD/SUB` and `MUL` is, and the smaller the constant is, then more clock cycles we can save on multiplication strength reduction.

In order to show how much cost it can reduce on real benchmark programs, such as PARSEC, we implement an llvm pass to insert two instruction counters: one is to count the cost of the original instruction and the other is to count the cost of optimized instruction. For each multiplication instruction, we instrument instructions to update the costs. Unfortunately, we have not gotten the PARSEC benchmark working yet.

## Our Thoughts

On most of the modern processors, the performance difference between add/subtract/shift operations and multiplication operations are not that huge. In a non-scientific computation workload, the optimization strength reduction for multiplication is negligible.

We also found that the strength reduction could be very useful on hardware design languages, such as Verilog for FPGA design. Since the multiplication circuit has a longer path than add/subtract/shift operations, strength reduction may be used to reduce the length of critical path and the total number of gates of a circuit.
