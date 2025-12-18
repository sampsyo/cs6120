+++
title = "TorchSplit: a Compiler Analysis for Graph Decomposition and Parallel Execution on Neural Networks"

[[extra.authors]]
name = "Jeffrey Qian"
[[extra.authors]]
name = "Ann Zhang"
+++

## Context and Problem Statement

Modern machine learning (ML) systems increasingly rely on multimodal models that combine multiple data sources into a single inference pipeline. These model types are extremely useful because of their flexibility, but they also pose an interesting systems challenge: how can we efficiently serve them at scale on multi-GPU hardware?

In practice, many deployments treat these models as monolithic black boxes. If a model is too large to fit on a single GPU, ML practitioners typically rely on model parallelism techniques such as pipeline parallelism (where different layers are executed on different GPUs), tensor parallelism (where a single layer is distributed across multiple GPUs), or data parallelism (where multiple replicas each process a minibatch). While effective, this approach ignores the internal structure of multimodal models, often leading to poor resource utilization and unnecessary bottlenecks.

In this blog post, we explore the idea that multimodal models are not monolithic—they are structured dataflow graphs with branches, joins, and components that have different compute and memory characteristics. If these components could be identified, separated, and scheduled independently, a serving system could exploit this component-level parallelism and allocate resources to alleviate bottlenecks. However, doing this is nontrivial: it requires extracting a model’s internal structure, determining which subgraphs can be safely split without changing semantics, understanding how each component’s performance scales with available GPU memory, and allocating components across GPUs in a way that improves end-to-end performance.

To address this challenge, we present TorchSplit, an end-to-end system that automatically decomposes PyTorch multimodal models into parallelizable components and optimizes their deployment across multi-GPU hardware. TorchSplit combines static dataflow analysis, profiling under memory constraints, and optimization-based resource allocation, and integrates with a production serving framework to execute the resulting deployment plan. The goal of TorchSplit is not to change model accuracy or architecture, but to improve serving throughput and latency by better leveraging existing hardware.

This post reports on the design, implementation, and evaluation of TorchSplit. We describe how the system extracts safe-to-split components from PyTorch models, profiles and allocates GPU resources to those components, and executes them in a distributed serving environment using Ray Serve. We then evaluate TorchSplit on a real multimodal model [(CLIP)](https://github.com/openai/CLIP.git) and show that componentized serving can significantly outperform a traditional monolithic deployment under high load.

## Implementation
### Architecture Overview
<img src="./2025-12-15-torchsplit/architecture_diagram.png" alt="architecture diagram" width="300"/>

TorchSplit is implemented as a tool that takes a PyTorch model as input and produces a componentized serving configuration. The overall architecture consists of three main modules: the **SplitClient**, the **Profiler and Optimizer**, and the **Runtime Environment**.

1. **SplitClient** - Provides a user-facing interface for defining models and generates input arguments for tracing and profiling.
2. **Profiler and Optimizer** - Performs dataflow analysis, runtime profiling, and solves an ILP allocation to generate an optimized execution plan.
3. **Runtime Environment** - Loads the execution plan and dispatches execution across the components.

### SplitClient: Model Tracing and Graph Generation 
The process begins by tracing the model’s forward pass using [Torch FX](https://docs.pytorch.org/docs/stable/fx.html), which converts the execution into FX graph intermediate representation (IR). This is a graph IR: each node corresponds to a specific (op)eration and edges represent the dataflow between operations through targets (variable names).

Initially, we planned to use proxy tensors (tensors with no dimensions) for tracing because they do not require a dataset or concrete inputs. However, we discovered that this approach does not support models with control flow (e.g., if/else statements). Depending on the IR export backend, tracing either fails with a `RuntimeError` or produces a layer-by-layer trace of a single execution path. As a result, we shifted toward using concrete tensors with known dimensions for tracing, which allows conditional values to be represented using phi nodes.

From this trace representation, TorchSplit constructs an internal directed acyclic graph (DAG) where nodes represent values and edges represent an op's dependencies. Although the resulting DAG represents the dataflow rather than control flow, the phi nodes preserve information about where values used downstream may originate from. This allows TorchSplit to analyze data dependencies across all possible branches, rather than only the path taken for a specific input. While this functionality is still under development, it will allow TorchSplit to determine whether the bodies of if/else blocks can be emitted as separate components. This is designed to support models with conditional branching in the future.

In the SplitClient, users can define the model and provide example inputs as such:
```python
class ClipClient(SplitClient):
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_model(self):
        return self.model

    def get_benchmarks(self, batch_size: int):
        # --- omitted code ---
        for model_args, model_kwargs in concrete_inputs:
            yield model_args, model_kwargs
```

### Profile and Optimizer: Graph Decomposition, Profiling, and ILP-based Allocation

To determine which parts of the model can be safely separated, TorchSplit analyzes the graph to identify Single Entry Single Exit (SESE) regions, defined by two nodes, A and B. To form a SESE region, A must dominate B (All paths from entry to B must go through A), and B must postdominate A (All paths from A to exit must go through B). In a dataflow graph, this property ensures that the interior of the region has no incoming edges from outside the region, allowing it to be extracted as an independent component with A acting as the input boundary and B as the output boundary. When exporting the final code, the TorchSplit sets A as the input node and B as the output node in the transformed FX graph module.

We compute dominators and post-dominators using the Lengauer–Tarjan algorithm and enumerate all O(n^2) candidate node pairs to identify valid SESE regions. Since many such regions may exist, the system greedily selects the largest disjoint regions, which typically correspond to meaningful model components such as modality-specific encoders. The remaining nodes form smaller coordination regions that handle data routing between components.

Once the model has been decomposed into components, TorchSplit profiles each component independently to characterize its performance under varying GPU memory constraints. For isolated profiling, the system replays the component execution in topological order. In the case of the CLIP model, TorchSplit identifies three components: A (vision encoder), B (text encoder), and C (merge step), with a linear execution order A, B, then C. Using the concrete inputs specified in the SplitClient, the profiler executes the forward pass of component A and records its output A′, followed by the forward pass of component B to obtain B′. The forward pass of component C is then executed using A′ and B′ as inputs. To avoid unnecessary GPU memory measurements, the intermediate tensors A′ and B′ are briefly paged to DRAM and transferred back to the GPU only when required for profiling component C.

To cap the memory, we use PyTorch’s [cuda API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.set_per_process_memory_fraction.html) to set the maximum memory available to the memory allocator. We do this for 1, 2, 4, 8, 10, 20, and 40 GB of memory (the target GPU is an A100 with 40GB of memory). The reason we picked these sizes is because they evenly divide 40GB, which makes it easier to allocate memory slices later on. For each memory size, we measure achievable throughput in terms of queries per second, qps, calculated from multiplying batch size by batches per second. We kept batch size fixed at 1 because of reasons mentioned later in the Future Work section. We also saw from this profiling step that 8GB appears to be sufficient for all components (More on this in the Evaluation section).

Given these profiles and a target multi-GPU environment, TorchSplit formulates resource allocation as an optimization problem. Each GPU can be partitioned into memory slices e.g. [8,8,8,8,8] GB slices for a 40 GB A100 GPU, each of which can host a replica of a component if the slice is large enough. The optimizer selects one memory layout per GPU and assigns component replicas to slices; the objective is to find the allocation, subject to resource constraints, which maximizes the minimum throughput across all pipeline components. The result is a concrete deployment plan that specifies how many replicas of each component to run and how GPU memory should be allocated. This ILP problem is formulated and solved using Gurobi via the [gurobipy Python API](https://docs.gurobi.com/projects/optimizer/en/current/reference/python.html). 

Finally, TorchSplit exports each selected component as an independent serialized PyTorch module, along with a context file that describes the global dataflow graph and execution plan. At serving time, a runtime loads only the components required for each replica. The system integrates with Ray Serve and uses GPU allocations derived from the ILP solver. 

Our open-source implementation is split across two repositories: one for [performing the split](https://github.com/jeffreyqdd/TorchSplit/) and one for [actually running the split model](https://github.com/az275/torchsplit_ray_deployment/).

## Evaluation

We performed our evaluation on the [CLIP](https://github.com/openai/CLIP) model, a multimodal vision and language model which is popular for image classification tasks. TorchSplit partitions the full CLIP model into 3 components: a vision encoder, text encoder, and merge step, which we label A, B, and C. A and B can run in parallel, while the paired outputs from both are the inputs to C. 

We ran experiments on one node of the [Perlmutter supercomputer](https://docs.nersc.gov/systems/perlmutter/architecture/) with 4 40GB NVIDIA A100 GPUs. We used the [HuggingFace food101](https://huggingface.co/datasets/ethz/food101) image classification dataset; each item consists of an image and a classification label which is converted to a text prompt. This image, text pair forms the input to the CLIP model. We deployed our models on Ray Serve. 

We measure the performance of a monolithic deployment of the CLIP model as our baseline. Specifically, this configuration places one copy of the full CLIP model on each of the 4 GPUs; this is the naive way of replicating across available resources, and is standard for model inference. 

The deployment configuration for the split CLIP model is somewhat more complicated. The goal is to efficiently pack components onto the available GPU resources to achieve better performance. We do this using an ILP solver following the procedure described in the previous section. A quick note -- for this project, we made a couple of simplifications: we hard coded the possible GPU memory layouts in our ILP problem to only allow [8,8,8,8,8] GB slices, since 8GB is the maximum amount of memory required by any of the CLIP model components (and some require less). This means that our allocation is actually less than optimal, and we can do even better. We also do not currently support batching in our actual deployment, since getting this to work proved somewhat difficult given time constraints. See some more comments on this in the ‘Future Work’ section. 

The allocation that this yields has 9 copies of component A, 8 copies of component B, and 3 copies of component C, each on an 8GB partition; intuitively, we can validate that this seems reasonable, since our profiling results showed that component C has the highest throughput. Ray allows us to manually configure model deployments via decorators, in which we can specify parameters such as the number of replicas and GPU resources for each component. 

We measure end-to-end latency, throughput, and GPU utilization across a few different send rates. We run each experiment for 5 seconds (5 * qps queries). We do not currently have batching on the server side, and process one query at a time. Latency is the time elapsed between when a client sends a query and when it receives the response; throughput is calculated as the total number of queries divided by the total time elapsed between when the first query is sent and the last response is received. 

<img src="./2025-12-15-torchsplit/throughput_vs_send_rate.png" alt="throughput vs send rate" width="300"/>

We observed that the componentized deployment does indeed outperform the monolithic baseline at higher send rates. At the lowest send rate (128 qps), both are able to reach throughput almost equal to send rate, meaning the servers are able to meet load requirements. The monolithic deployment is unable to sustain a throughput of more than around 165 queries per second; the componentized deployment hits its limit at about 260 queries per second, about 57% higher. 

<img src="./2025-12-15-torchsplit/latency_vs_send_rate.png" alt="latency vs send rate" width="300"/>

The componentized deployment also achieves much lower latencies at higher send rates than the monolithic one. The bulk of these multi-second latencies is queueing time; the actual model runtime for a single request is on the order of 10ms. Even with the stage-to-stage handoffs required in the componentized deployment, it performs much better. 

Finally, we found that the componentized deployment achieves better GPU resource utilization. We queried GPU utilization % and memory usage statistics for each GPU (using nvidia-smi) every 100ms while the program was running. The monolithic deployment averaged about 25% utilization and 1215 MiB memory; the componentized one attained 46% utilization and 5100 MiB memory. There’s definitely still a lot of room for improvement here; this goes back to optimizing the allocation. 

Disregarding the time spent profiling, TorchSplit takes about 7.7 seconds on CLIP which has 943 nodes. The major components of this are:
1. Torch FX Graph Export: 2.77s
2. Dominance Analysis and SESE Extraction: 2.65s
3. Export: 0.51s 

The rest of the the execution is spent on Python module imports. This was obtained via (py-spy)[https://github.com/benfred/py-spy], a sampling-based Python profiler. The ILP solver was run separately, and the time is negligible since the problem size is very small.

## Future Work

There are a lot of things we can do to further improve performance; we’ll talk about a couple which we touched on previously. 

First, we can do a better job of allocating components to GPUs. This would involve incorporating more information from the profiling step into the allocation problem. In particular, we should support batching; many ML models scale well with batching, so specifying inputs to the ILP based on throughput and GPU utilization at different batch sizes could generate a better allocation. Batching is difficult because Torch FX yields static graphs (fixed to a single batch size), so we would either need to export multiple graphs at different sizes or modify the graph to support variable batch sizes.

[Multi-instance GPU partitioning](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/) -- partitioning a single GPU into isolated instances to eliminate interference between models running on the same GPU -- and finer grained GPU profiling could also help us get to better resource utilization and performance. 

Stage to stage handoffs introduce overheads in a componentized deployment which do not exist for a monolithic application, so minimizing them is important. Ray stage to stage handoffs are done via TCP; this is slow, and different model serving platforms may offer the opportunity to use RDMA. Depending on the hardware platform, NVLink might also be possible. 

## GenAI Statement

We used ChatGPT to generate the Python plotting scripts. It’s quite good at this.

We also used ChatGPT to help diagnose issues with the Ray deployment; this involved pasting the error messages from the Ray logs along with a brief description of the problem/context. It’s not so good at this; it was sometimes helpful and sometimes completely useless. 
