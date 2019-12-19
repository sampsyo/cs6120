 +++
title = "Quantum Vectorization"
extra.bio = """
  [Dietrich Geisler](https://www.cs.cornell.edu/~dgeisler/) is a 3rd year PhD student researching Language Design and Compilers.  Enjoys gaming and climbing.
  Philip Bedoukian is a 3rd year PhD student in ECE. His research focuses on reconfigurable hardware.
"""
extra.author = "Dietrich Geisler"
extra.author = "Philip Bedoukian"
+++

The code used in this blog post is hosted [here](https://github.com/pbb59/ScaffCC).

## Introduction

In this blog, we describe our efforts to develop a compiler pass to vectorize the implicit parallelism present in quantum algorithms. Quantum algorithms are probabilistic, and so need to be run multiple times to get a "reliable" result. Since each of these program runs are independent, several can be performed simultaneously on the same hardware without changing the final result, so long as the hardware has space to support the additional logic.  

In this project, we developed an LLVM pass to transform code to help take advantage of this program structure.  Our LLVM pass rewrites code to duplicate all algorithm instructions associated with each array onto physical hardware.  We cannot conclude if this approach provides speedup without a proper experimental setup, but we have found that such a pass can be run on realistic quantum code to produce somewhat vectorized algorithms.


## Quantum Computing

Quantum Computing has exploded into the popular imagination in the past decade due to the promise of massive theoretical speedups over conventional digital computers. Whether real quantum hardware can live up to the promise remains to be seen, but that has not stopped researchers from developing complex toolflows and algorithms.

The computing paradigm of quantum computers is inherently different from a standard "classical" computer. Instead of representing a bit as either a `0` or `1`, quantum bits (qubits) represent a bit of information using a quantum superposition `a |0> + b |1>`, where `|0>` and `|1>` represent possible realizable states and `a` and `b` are normalized constants related to probability of measuring the respective state. Although `a` and `b` theoretically hold infinite information, it is only practically possible to measure a bit of information from the state as in the classical case as the state collapses to one of the realizable state `|0>` or `|1>` upon measurement.

Quantum computing offers unique computing properties due to the nature of a qubit. The main computational differences between quantum and classical computing include the following properties:

|  Property       |  Quantum    | Classical CPU |
| ------  | ------------- | ------------------- |
| Architecture   | Spatial | von Neumann  |
| Data    | Quantum State | Voltage Level |
| Control | External (Laser) | Voltage Level |
| States per bit | Exponential | Linear |

In general, a quantum computer can implement all of the computation primitives that a classical one can. Both, in theory, can be turing complete with a universal set of logic gates. However, quantum computing also has computational primitives that classical computers don't share. These primitives are key to quantum supremacy: the concept that a quantum computer can theoretically outperform classical computers.

| Unique compute | Example Usage |
| ----------------| ------------- |
| Large State Space  | Chemical Reaction Simulation |
| Entanglement | Combinatorial Optimization (TSP) |
| Amplitude Magnification | Database Search |
| Probabilistic (multiple results) | ? |
| Phase | ? |

A potential downside to quantum computing is that it is inherently probabilistic. An output on one execution may be entirely different from the output on the next run. Quantum algorithms must be designed so that the correct answer must have measurement probability >50%. The answer can then be inferred by repeating the execution many times and taking the majority result. Many quantum algorithms exist in Bounded Quantum Polynomial class (BQP) where the correct answer can be found in polynomial time with probability at least 2/3.  It can, however, practically be time and resource intensive to run quantum programs a sufficient number of times to achieve a reasonable confidence.

## Opportunities for Vectorization

The probabilistic nature requires that multiple repeated runs of the same program be executed. The number of runs required to obtain a "correct" result depends on one's error threshold and the design of the algorithm; specifically, the number of runs to obtain error `e` is given as `O(log(1/e))`. Thus, there are diminishing returns for running the algorithm many times, but it is important to run the algorithm a "reasonable" amount to achieve acceptably low error.

The naive method to repeatedly apply the algorithm is to run many iterations of the algorithm sequentially. This serialization can potentially increase the runtime depending on how many repeated applications are required. Consider, for instance, the entanglement program below, which must be run a large number of times to produce a correct result:

```C
module catN ( qbit *bit, const int n ) {
  H( bit[0] );
  for ( int i=1; i < n; i++ ) {
    CNOT( bit[0][i-1], bit[0][i] );
    CNOT( bit[1][i-1], bit[1][i] );
  }
  MeasZ(bit);
}

void main () {
  qbit bits[4];
  catN( bits, 4 );
}

```

By preempting the need to run this program multiple times, we can directly incorporate the implicit outer loop and produce something like the following:

```C
module catN ( qbit *bit, const int n ) {
  H( bit[0] );
  for ( int i=1; i < n; i++ ) {
    CNOT( bit[0][i-1], bit[0][i] );
    CNOT( bit[1][i-1], bit[1][i] );
  }
  MeasZ(bit);
}

void main () {
  qbit bits[2][4];
  for (int i = 0; i < 2; i++) {
    catN( &(bits[i]), 4 );
  }
}
```

Now that we have a data-parallel outer loop, we can schedule multiple runs together in spare quantum resource and potentially vectorize the runs if the architecture allows. By exposing this parallelism, we expect to achieve speedup over running the repeats sequentially. Note that we are _not_ trying to vectorize non-data structure in the underlying algorithm; such an implementation would require more information about each algorithm and may fail due to data dependencies. We are instead vectorizing the implicit data-parallel nature of data structures in probabilistic computing.

## Implementation

We designed a quantum compiler pass within the [ScaffCC](https://github.com/epiqc/ScaffCC) compiler infrastructure. ScaffCC adds IR passes and quantum computer backends on top of LLVM, so our pass is written as one would for a classical compiler.

The pass first records each instance of the `alloca` command for vectorization. We make the assumption that qubit arrays are the only memory structures allocated by these programs.  This assumption is based on observations of program samples included in the ScaffCC repository.

Once every allocation is recorded, each of these commands is cloned a number of times equal to the `qvlen` argument. We then fully traverse the dataflow graph to copy all dependent instructions. We traverse the dependence graph starting from the allocations in a breadth-first manner, so that we copy a dependent instruction only when all of its dependencies have already been copied. This is required to have the copied values available for use in later instruction copies. Quantum computers are spatial architectures, so functions are inlined to a single basic block. Thus, our dataflow graph algorithm was able to reach the whole program.

We do not actually implement vector instructions because it would require extensive backend work to target the simulator. The simulator does support operations in parallel, but does not have give any timing information. Because of this, the extensive backend work would also not show any meaningful results.

It is worth noting that this implementation does not scale to situations where qubit allocations include dependencies (such as if a qubit allocation used the size of a previous allocation).  We choose to ignore these cases as a simplifying assumption.

## Evaluation

We evaluated our technique using the ScaffCC compiler infrastructure and a [quantum computer simulator](http://qutech.nl/qx-quantum-computer-simulator/). Due to constraints of the simulator we were limited to small benchmarks using a small number of qubits. We still chose to use the simulator to check for correctness as well as get a sense of the probability distributions for the simulated algorithm. We identified six benchmarks that had ~10 or less qubits. One of the benchmarks, QFT (quantum Fourier transform), is an intermediate step in most algorithms and not meant to be measured, so we excluded it. The benchmarks are enumerated below, along with the number of times to repeat the execution. This number is mostly made up and is between 10 and 100 depending on how fast the algorithm ran.

We used a pass to count the number of dynamic gate operations for each of the benchmarks. Note that all loops are unrolled in a quantum program because quantum computers are a spatial architecture, i.e., the number of static instructions is the same as the number of dynamic instructions.

| Benchmark | Qubits Used | Gates | Repeats
| ----------------| ------------- | --------- | -------------- |
| Cat | 4 | 8 | 100 |
| Ising | 10 | 220 | 100 |
| VQE | 4 | 148 | 100 |
| Grover | 11 | 174 | 100 |
| Ground State | 6 | 8713 | 10 |

The simulator does not give timing information, so we created a rough timing model. We assume the target quantum computer uses Ion Trap technology. Here, a microwave laser can implement quantum gates by shining onto qubits. SIMD is possible by directing the laser to multiple qubits at once ([citation](https://dl.acm.org/citation.cfm?id=2694357)). These "instructions" are likely not as fast as a >1GHz instruction cache on a classical computer, so amortizing the cost of control is important. Thus, we quantify the timing by the number of total laser pulses required to run the algorithm with enough repeats.
Additionally, we consider a quantum computer with 20 logical qubits that can be used for multiple simultaneous runs.

We do not consider any spatial scheduling problems and assume the qubit regions working on different runs are effectively in isolation. We can statically predict the best run-time using our model,

`Time = Repeats * Gates * Gate_Time / Vector_Length`

We consider relative speedup to baseline, so the actual time to execute a gate is a constant factor that will be divided out. The theoretical speedup is then given by,

`Speedup = Floor(Max Qubits / Used Qubits)`

Our theoretical results for each benchmark on a quantum computer is given below for a 20 qubit machine and a 53 qubit machine like Google's recent Sycamore computer, which achieved quantum supremacy for the first time.

| Benchmark | Speedup (20 qubits)  | Speedup (53 qubits)
| ----------------| ------------- | -------------- |
| Cat | 5 | 13 |
| Ising | 2 | 5 |
| VQE | 5 | 13 |
| Grover | 1 | 4 |
| Ground State | 3 | 8 |

We also experimentally compile each benchmark with our pass and execute the program through the simulator to check for "correctness". Each program successfully compiled and executed on the simulator. We explicitly checked the `cat` program for correctness. In this algorithm a group of 4 qubits are entangled to be either all 0s or all 1s. We verified that there were multiple groups of 4 qubits with this property. The other algorithms also seemed to have reasonable outputs (a mix of 0s and 1s that changed on a run-by-run basis).

## Conclusion

We implemented an LLVM pass to vectorize the implicit data parallel repetition loop needed to produce precise quantum computing results.  Through this implementation, we show that such an optimization is possible and can be readily applied to some common quantum programs.  We then used this pass with a quantum gate simulator to predict the speedup possible by applying such an optimization.
