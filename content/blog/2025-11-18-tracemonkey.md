+++
title = "Retracing The Tracing JIT"
[extra]
bio = """
  Jeremy Ku-Benjet writes code to make it easier to write code.
  Sunwoo Kim is pursuing PhD in Electrical and Computer Engineering at Cornell University. He's researching AI-assisted methods to design theoretically optimal computing systems.
"""
[[extra.authors]]
name = "Jeremy Ku-Benjet"
link = "https://jeremykubenjet.com"
[[extra.authors]]
name = "Sunwoo Kim"
link = "sunwookim028.github.io"
+++

## Background and Significance
As dynamic languages like JavaScript became ubiquitous in client-side web programming, executing them efficiently became a critical challenge for browser vendors. The lack of static types meant that ahead-of-time compilers were forced to emit slow, generic code, while static analysis proved too expensive for the interactive constraints of the web.

Virtual machines (VMs) eventually adopted mixed-mode execution, jumping back and forth between an interpreter and a Just-In-Time (JIT) compiler. While traditional JITs operated at the granularity of functions (methods), **TraceMonkey**-the system proposed in this paper—focused on the program path during runtime, crossing function boundaries.

TraceMonkey identified hot loop traces at runtime and recorded instructions as they were executed. This allowed the VM to generate type-specialized machine code for subsequent loop iterations. By doing so, the system could specialize code aggressively and achieve function inlining implicitly.

Although TraceMonkey was eventually discontinued as web workloads shifted toward patterns less favorable to tracing, it remains historically significant. It was the first production tracing JIT in a major browser (Firefox) and pioneered the tracing paradigm in the wild.

## Key Ideas & Contributions
The primary contribution of this paper was the efficient tracing mechanism implemented in TraceMonkey. It handled deep nested loops effectively and demonstrated valid speedups (2x–20x) over the baseline SpiderMonkey interpreter on the SunSpider benchmark.

**How it Works:**
1. Interpretation: TraceMonkey starts as a bytecode interpreter.
2. Recording: It operates at the granularity of individual loops. When a loop becomes "hot," the system enters recording mode.
3. LIR Generation: A sequence of operations is recorded in a Static Single Assignment (SSA) Low-Level IR (LIR).
4. Guards: Runtime checks—known as guards—are inserted before branches and type specializations.
If a guard fails, execution "side exits" back to the interpreter or to a different trace. If a side exit becomes hot, a new trace is recorded from that point. Divergent branches are recorded as branch traces, eventually forming a trace tree that covers multiple hot paths through the loop.

### The Nested Loop Solution

A naive tracing JIT struggles with nested loops. If an inner loop has multiple paths, the outer loop might be recorded multiple times (once for every exit), leading to tail duplication and exploding code size.

TraceMonkey introduced an algorithm to recognize inner loop headers. Instead of inlining the inner loop into the outer loop's trace, it treats the inner loop as a separate trace tree. The outer loop’s trace simply "calls" the inner loop’s trace. This modular composition prevents exponential trace growth.


## Diamonds in Loops
For straightline code, these algorithms works well, but they cause cause trouble in the case of branches.A simple case of this is a branch inside of a loop, causing a diamond in the control flow. If both edges are considered hot, a low bar of two traversals for TraceMonkey, both edges will be stitched together into a single trace. Was this worthwhile?

We don't have numbers to show this, but consider the following thought experiment: first, what if each branch was taken a similar amount of times. In this case, due to TraceMonkey's quick baseline compiler, nanojit, this compilation should pay off. However, consider the other case where one branch is just barely hot. In this case, the cycles spent compiling the trace would never pay off as the trace would not be used enough. And a case of a branch like this isn't entirely strange, for example a piece of code in a loop which may infrequently but periodically append to a log.

Now consider if there were multiple of these diamonds sequenced. This might be the case if there are sequenced method calls as the JIT needs runtime information to dispatch the correct method. When running these instructions, TraceMonkey will start by creating a trace for one path. However, assuming the branch conditions are not stable, a different path will be chosen. Depending on how unstable each branch condition is, it may be unlikely to traverse a set of conditions and have the same results of each condition, but probably not so rare that a sequence of conditions would occur twice to get a trace compiled. This would cause both many exits from traces and many compiled traces, a large performance penalty.

A way to view the problems presented in these examples is there is too low a bar for classifying control flow edges as hot. The first example shows wanting for a better criterion when branch conditions are stable and the second example shows the same when branch conditions are unstable. TraceMonkey, a "pure" tracing JIT, leans so heavily into traces as its technique to gain performance it needs to eagerly create these traces but doing so can lead to performance regressions when compared to other JITs.


## Speculation
### Doesn't Hardware Also Speculate About Branches?
An immediate reading of TraceMonkey's implementation brings to mind, if only by the name "speculation," speculative execution in modern processors. There are similarities: both make predictions about the results of future branches, or in the case of TraceMonkey often types, to get a performance increase. Though, past this, the differences waver. When a processor encounters a branch, it will predict the outcome of that branch before the branch condition can execute and speculatively executing instructions based on that prediction. If the prediction is correct, the processor will commit it's changes, else it will squash the changes and execute based on the correct path. Alternatively, when TraceMonkey encounters a branch in a trace, if the result of the branch is what TraceMonkey predicted, and thus compiled into the trace, TraceMonkey will continue execution, else it will fall back to the interpreter. The main difference here is while TraceMonkey is speculating about conditions on previously run code so it can guess what to compile, it never speculatively executes instructions.

With this difference on what is being speculated, we'd argue speculation in processors and speculation in TraceMonkey are orthogonal processes, meaning the scary security implications of speculatively executing code, but also the fun parallelism and cheaper cost of a branch missprediction don't apply.

### How Bad Are Mispredictions Anyway?
Returning back from traces to interpreted code can be quite costly. Upon a guard failure it requires copying all of the data modified during trace execution to the interpreter's structures and then returning to running the program through the interpreter. Given this, the (performance) tradeoff of speculation can be simplified to be between the extra cost of recording and returning from the trace and the time saved. This is a worthwhile trade off if the compiler saves more time than it looses to these extra costs. The thing to notice is both of these costs grow significantly when TraceMonkey misspredicts a branch or type: the code must exit and then it will likely start another trace. TraceMonkey's trace trees stitch together traces which help mitigate runtime penalties from these misspredictions, but mitigations don't make these penalties go away.

A point of comparison is with the [speculation in WebKit's JavaScriptCore](https://webkit.org/blog/10308/speculation-in-javascriptcore/) JIT. This isn't a perfect comparison as while both of these JITs are speculating, JavaScriptCore seems to do more onerous computation during it's compilation and often acts on functions instead of traces, also which JavaScriptCore seems more eager to throw away than TraceMonkey, causing a higher cost of misspredicts. However, the drastic differences between the two compilers' philosophies towards speculation still makes this useful.

The conclusion the engineers behind JavaScriptCore came to was misspredicts are so costly compared to the baseline compiler and interpreter that without a prediction rate of nearly 100%, predictions were not worthwhile. This is reflected in the points system JavaScriptCore uses to determine if it should transition to a heavily speculative JIT: A function has to be called ~70 times or run ~1000 loop iterations to be compiled using a tier which performs heavy speculation. This is in contrast to TraceMonkey which will begin compiling a trace after 2 runs.

TraceMonkey's eager speculation sometimes really pays off, clear in a couple of the test cases its authors present, however it's strategy is risky. This has caused current day JavaScript JITs to move away from this model. TraceMonkey itself has been replaced by [IonMonkey (now WarpMonkey)](https://doi.org/10.1109/CGO.2013.6495006), which looks similar to (and looks to predate) JavaScriptCore's more conservative speculation model. History judged TraceMonkey's speculation overeager.


## The Spectrum of JITs
### Why Couldn’t TraceMonkey Survive?
In our class discussion, a common question was whether TraceMonkey—or tracing JITs in general—are still used. To answer this, we have to look at the spectrum of compilation granularity.

- Method-based JITs: Operating at the opposite end of the spectrum, these compile entire functions at once (e.g., Google’s V8, Apple’s SquirrelFish Extreme). They allow for traditional static optimizations but require complex analysis for dynamic typing.
- Tracing JITs: These focus on specific paths. Their critical weakness is control flow divergence. To cover a loop with many branches, a tracer must record a new branch trace for every divergent path, leading to code cache explosion.

TraceMonkey attempted to "blacklist" loops that frequently aborted, but real-world web workloads proved to be highly branchy rather than type-stable. This violated the core assumptions that make tracing efficient.

Modern JITs eventually adopted a tiered approach. While TraceMonkey's recording phase blocked execution (running at ~1/200th interpreter speed), modern engines use up to four tiers of optimization to balance startup time with peak performance. They use background threads for recompilation as well, as mentioned in this paper as a future work.
