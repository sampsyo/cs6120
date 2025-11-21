+++
title = "TBD"
[extra]
bio = """
  TBD
"""
[[extra.authors]]
name = "TBD"
link = "example.com"  # Links are optional.
[[extra.authors]]
name = "TBD"
+++
## Speculation
### Doesn't Hardware Also Speculate?
An immediate reading of TraceMonkey's implementation brings to mind, if only by the name "speculation," speculative execution in modern processors. There are similarities: both make predictions about the results of future branches, or in the case of TraceMonkey often types, to get a performance increase. Though, past this, the differences waver. When a processor encounters a branch, it will predict the outcome of that branch before the branch condition can execute and speculatively executing instructions based on that prediction. If the prediction is correct, the processor will commit it's changes, else it will squash the changes and execute based on the correct path. Alternatively, when TraceMonkey encounters a branch in a trace, if the result of the branch is what TraceMonkey predicted, and thus compiled into the trace, TraceMonkey will continue execution, else it will fall back to the interpreter. The main difference here is while TraceMonkey is speculating about conditions on previously run code so it can guess what to compile, it never speculatively executes instructions.

With this difference on what is being speculated, I'd argue speculation in processors and speculation in TraceMonkey are orthogonal processes, meaning the scary security implications of speculatively executing code, but also the fun parallelism and cheaper cost of a branch missprediction don't apply.

### How Bad Are Misspredictions Anyway?
As mentioned above, traces are great, but returning from them back to interpreted code can be quite costly. Upon a guard failure it requires copying all of the data modified during trace execution to the interpreter's structures and then returning to running the program through the interpreter. Given this, the (performance) tradeoff of speculation can be simplified to be between the extra cost of recording and returning from the trace and the time saved. This is a worthwhile trade off if the compiler saves more time than it looses to these extra costs. The thing to notice is both of these costs grow significantly when TraceMonkey misspredicts a branch or type: the code must exit and then it will likely start another trace. TraceMonkey's trace trees stitch together traces which help mitigate runtime penalties from these misspredictions, but mitigations don't make these penalties go away.

A point of comparison is with the [speculation in WebKit's JavaScriptCore](https://webkit.org/blog/10308/speculation-in-javascriptcore/) JIT. This isn't a perfect comparison as while both of these JITs are speculating, JavaScriptCore seems to do more onerous computation during it's compilation and acts on functions instead of traces, which JavaScriptCore seems more eager to throw away than TraceMonkey, causing a higher cost of misspredicts. However, the drastic differences between the two compilers' philosophies towards speculation still makes this useful.

The conclusion the engineers behind JavaScriptCore came to was misspredicts are so costly compared to the baseline compiler and interpreter that without a prediction rate of nearly 100%, predictions were not worthwhile. This is reflected in the points system JavaScriptCore uses to determine if it should transition to a heavily speculative JIT: A function has to be called ~70 times or run ~1000 loop iterations to be compiled using a tier which performs heavy speculation. This is in contrast to TraceMonkey which will begin compiling a trace after 2 runs.

TraceMonkey's eager speculation sometimes really pays off, clear in a couple of the test cases its authors present, however it's strategy is risky. This has caused current day JavaScript JITs to move away from this model. TraceMonkey itself has been replaced by [IonMonkey (now WarpMonkey)](https://doi.org/10.1109/CGO.2013.6495006), which looks similar to (and looks to predate) JavaScriptCore's more conservative speculation model. History judged TraceMonkey's speculation overeager.
