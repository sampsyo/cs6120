 +++
title = "Quantum Vectorization"
extra.bio = """
  Dietrich Geisler is a person.
  Philip Bedoukian is a 3rd year PhD student in ECE. His research focuses on reconfigurable hardware.
"""
extra.author = "Dietrich Geisler"
extra.author = "Philip Bedoukian"
+++

The code used in this blog post is hosted [here](https://github.com/pbb59/ScaffCC).

## Introduction

In this blog, we describe our efforts to develop a compiler pass to vectorize the implicit parallelism present in quantum algorithms. As quantum algorithms are probabilistic they need to be run multiple times to get a ``reliable'' result. Our pass effectively copy and pastes the algorithm multiple times onto physical hardware in order to share control between the runs.

## Quantum Computing

Quantum Computing has exploded in popularity in the past decade due to the promise of massive theoretical speedups over conventional digital computers. Whether real quantum hardware can live up to the promise remains to be seen, but that has not stopped researchers from developing complex toolflows and algorithms.

The computing paradigm of quantum computers is inherently different from a standard "classical" computer. Instead of representing a bit as either a `0` or `1`, quantum bits (qubits) represent a bit of information using a quantum superposition `a |0> + b |1>`, where `|0>` and `|1>` represent possible realizable states and `a` and `b` are normalized constants related to probability of measuring the respective state. Although, `a` and `b` technically hold infinite information, one can only measure a bit of information from the state as in the classical case. This is because the state collapses to one of the realizable state `|0>` or `|1>` upon measuring.

Quantum computing offers a lot of unique computing properties due to the nature of a qubit. The main differences between quantum and classical computing are enumerated below.

|  Property       |  Quantum    | Classical CPU |
| ------  | ------------- | ------------------- |
| Architecture   | Spatial | Von-Neumann  |
| Data    | Quantum State | Voltage Level |
| Control | External (Laser) | Voltage Level |
| States per bit | Exponential | Linear |

In general, quantum computer can implement all of the computation primitives that a classical one can. Both, in theory, can be turing complete with a universal set of logic gates. However, quantum computing also has computational primitives that classical computers don't share. These primitives are key to quantum supremacy: the concept that a quantum computer can theoretically outperform classical computers.

| Unique compute | Example Usage |
| ----------------| ------------- |
| Large State Space  | Chemical Reaction Simulation |
| Entanglement | Combinatorial Optimization (TSP) |
| Amplitude Magnification | Database Search |
| Probabilistic (multiple results) | ? |
| Phase | ? |

A potential downside to quantum computing is that it is inherently probabilistic. An output on one execution may be entirely different from the output on the next run. Quantum algorithms must be designed so that the correct answer must have measurement probability of at least >50%. The answer can then be inferred by repeating the execution many times and taking the majority result. Many quantum algorithms exist in Bounded Quantum polynomial class (BQP) where the correct answer can be found in polynomial times with probability of at least 2/3.

## Opportunities for Vectorization

The probabilistic nature requires that multiple repeated runs of the same program be executed. The number of runs required to obtain a "correct" result depends on one's error threshold and the design of the algorithm. The number of runs to obtain error `e` is given as `O(log(1/e))`. Thus, there is diminishing returns for running the algorithm lots of times, but it is important to run the algorithm a "reasonable" amount to achieve low error.

The naive method to handle the repeated runs is to run many iterations of the algorithm sequentially. This serialization can potentially increase the runtime depending on how many repeated runs are required. However, these repeated runs can be considered as a part of the original program. The entanglement program below illustrates this.

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

Instead we can incorporate the implicit outer for-loop in the program and produce something like the following.

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

Now that we have an data-parallel outer loop, we can schedule multiple runs together in spare quantum resource and potentially vectorize the runs if the architecture allows. By exposing this parallelism, we expect to achieve speedup over running the repeats sequentially. To be clear, we are not trying to vectorize code in the original algorithm; that would be much more challenging due to potential data dependencies. We are vectorizing the implicit data-parallel nature of probabilistic computing.

## Implementation

We designed a quantum compiler pass within the [ScaffCC](https://github.com/epiqc/ScaffCC) compiler infrastructure. ScaffCC adds IR passes and quantum computer backends on top of LLVM, so our pass is written like one would for a normal classical compiler.

TODO what does the pass actually do





## Evaluation

We evaluated our technique using the ScaffCC compiler infrastructure and a [quantum computer simulator](http://qutech.nl/qx-quantum-computer-simulator/). Due to constraints of the simulator we were limited to small benchmarks using a small number of qubits. We still chose to use the simulator to check for correctness as well as get a sense of the probability distributions for the simulated algorithm. We identified six benchmarks that had ~10 or less qubits. One of the benchmarks QFT (quantum fourier transform) is an intermediate step in most algorithms and not meant to be measured, so we excluded it. The benchmarks are enumerated below, along with the number of times to repeat the execution. This number is mostly made up and is between 10 and 100 depending on how fast the algorithm ran.

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

TODO somehow quantify what we actually did?

## Conclusion

We implemented a quantum compiler pass to vectorize the implicit data parallel outer loop in quantum programs. 


