+++
title = "LaTeX Equation Language (LEL): LaTeX equation to LLVM"

[extra]
latex = false
bio = """
Ruolin Ye is a student in Advanced Compilers.
"""

[[extra.authors]]
name = "Ruolin Ye"
+++

# Intro
LaTeX is a common language for writing professional documents like academic papers. In the past, I have worked on many research projects that involve implementing mathematical equations. It’s natural to think of a tool that converts the LaTeX formatted equations into usable code directly and allows the generated code to connect with other developing technologies.

# Problem Statement
This project asks: *can we automatically turn real LaTeX equations into executable code via a small compiler?*  
I build a pipeline that:
- normalizes LaTeX math from real sources (using MiTeX),
- parses a LaTex Equation Language (LEL) supporting arithmetic, conditionals, and function calls, and
- compiles each accepted equation to LLVM IR and checks it against a reference evaluator.

By design, LEL only targets **scalar** expressions with simple function headers (`name(args) = expr`), and explicitly does *not* attempt to handle the full complexity of LaTeX: integrals, matrices, sums/products, asymptotic notation, or rich macro systems are all out of scope for this first prototype.

# Implementation
I will detail the implementation in this section. I will first introduce the datasets I use, explain the components and compare what I proposed vs. what's actually implemented, and then detail how I implemented each component.
## Dataset
### A tiny hand-written set
A curated suite of 24 hand-written LEL functions covering arithmetic, powers, fractions, zero-argument functions, conditional expressions, and calls. 
### Hugging face latex-formulas dataset 
`latex-formulas` is a large-scale dataset of images paired with corresponding LaTeX equations. I downloaded the cleaned Parquet shards from the Hugging Face [dataset](https://huggingface.co/datasets/OleehyO/latex-formulas/tree/main/cleaned_formulas)`latex-formulas` and extracted the `latex_formula` column to build a text-only corpus. This gives me 552,339 distinct equations.
### Equations from ArXiv
For the arXiv part of the evaluation, I built a corpus by querying the cs.RO (robotics) category for 2,989 papers, downloading their LaTeX source tarballs, and scanning all .tex files for math environments such as equation and align. Each extracted equation is stored as a JSONL record containing the arXiv ID, source file path, environment type, and raw LaTeX string. This gives me a realistic set of 42,896 in-the-wild equations.

## Components Implemented
I tried my best to implement most of the components in the proposal but have to pivot on some of the festures. I summarize the components as follows:
### Fully achieved
- **LEL language design** for:
  - arithmetic expressions
  - function definitions
  - function calls
  - conditional expressions via ternary `cond ? then : else`
- **LLVM backend** that compiles LEL functions into executable code.
- **End-to-end evaluation framework** with:
  - generated C test drivers
  - compilation and execution via LLVM/Clang
  - correctness checking on two real corpora (robotics equations and HF LaTeX formulas).
### Partially achieved / simplified
- **Conditionals** are supported at the LEL level via the ternary operator (`cond ? then : else`), but are **not** directly parsed from LaTeX `\begin{cases} ... \end{cases}` blocks.
- **Recursive functions** are supported in the language and IR (we can emit recursive LLVM functions), but are treated as **unsupported** in the evaluation pipeline to avoid non-terminating tests (no end-to-end recursion results).
- **Argument inference from the RHS** (collecting free variables as parameters) from the original proposal was **dropped** in favor of explicit function headers of the form `name(args) = expr` or `name = expr`.
### Minor type change
- The implementation uses a single `double` type for all parameters and return values, instead of the integer type originally mentioned in the proposal.

## Implementation Details
### MiTeX-based normalization
Raw LaTeX equations from both corpora are first normalized using MiTeX. I treat each equation as a tiny standalone LaTeX document by wrapping it in a minimal preamble `(\documentclass{article}, \usepackage{amsmath})` and placing the equation in a display math environment. For each such document, I call the mitex compile CLI to produce a Typst source file. I then apply a simple line-based heuristic to extract the first non-directive line that looks like math content and store it as a typst_equation string in JSONL. This step standardizes a wide variety of LaTeX syntaxes into a more regular, MiTeX-normalized format while preserving the symbolic structure (identifiers, operators, parentheses). Only equations where MiTeX successfully returns Typst are passed on to the LEL front-end.
### LEL language and parser
On top of these normalized equations, I define the LaTeX Equation Language (LEL), a deliberately small expression language that I can reliably parse and compile. A LEL program consists of a single function definition of the form `name(args) = expr` or `name = expr`, where name and each parameter follow a simple identifier pattern `[A-Za-z][A-Za-z0-9_]*`. The expression grammar includes arithmetic `(+, -, *, /)`, exponentiation (`^`, right-associative), unary minus, parentheses, variables, numeric literals, and function calls such as `f(x)` or `g(x, y)`. I implement a tokenizer that converts a Typst math string into a flat token stream (IDENT, NUMBER, SYMBOL) and a hand-written recursive-descent parser that produces a small AST with node types like `Var`, `Const`, `UnaryOp`, `BinOp`, and `Call`.

To support conditional expressions, I extend the expression grammar with comparison operators `(<, >)` and a ternary operator `cond ? then : else`. In the AST this is represented by `Compare` and `IfExpr` nodes, which allow LEL to express piecewise behavior such as `relu(x) = (x > 0) ? (x) : 0` or `abs(x) = (x > 0) ? (x) : (0 - x)`. The top-level parser first splits the equation at the = sign, parses the left-hand side as a function header (name and parameter list), then tokenizes and parses the right-hand side into an expression AST. Any equation that does not fit this function-definition shape, or that uses constructs outside the grammar (for example integrals, matrices, or Typst macros like `mitexsqrt`), is rejected and excluded from the LEL corpus.
### LLVM IR code generation
Given a parsed FunctionDef AST, I generate LLVM IR for a single function with signature
```
double name(double, double, ...);
```
using one double parameter per formal argument and a double return type. The code generator recursively lowers each expression node into LLVM instructions. Binary arithmetic nodes `(+, -, *, /)` become the corresponding floating-point operations (fadd, fsub, fmul, fdiv); exponentiation ^ is implemented as a call to the C library pow function. Variables are mapped to the incoming function parameters, and numeric literals become double constants. Function calls in the AST are lowered to direct LLVM call instructions with the same double signature. For this project I treat all calls as calls to external functions; the test harness provides C stubs for these functions so that the IR always links successfully.

Conditional expressions compile down to standard SSA control flow. For an IfExpr node, I generate a comparison (fcmp on the Compare operand), a conditional branch to "then" and "else" basic blocks, and a join block with a phi node that merges the two branch values into a single result. The overall code generator builds a minimal control-flow graph per function and emits IR as text, which I then assemble and optimize using the standard LLVM toolchain (llc followed by clang) to produce an object file and, finally, a native executable when linked with a C driver.
### Reference evaluator and C test harness
To test correctness end-to-end, I pair the LLVM backend with a simple reference evaluator and automated C drivers. The reference evaluator is a pure Python interpreter over the same AST used for code generation: it implements evaluation rules for Const, Var, UnaryOp, BinOp, Compare, and IfExpr, using Python float for all values. Calls are handled via a conservative stub semantics: when the evaluator encounters a Call, it either returns the first argument or raises an "unsupported" error if the callee would require nontrivial semantics. I also detect self-recursive functions syntactically and mark them as unsupported in the evaluator to avoid non-terminating tests.

For each LEL function that passes parsing and is supported by the reference evaluator, the evaluation script synthesizes a small set of test inputs by sampling a Cartesian grid of positive double values for the parameters. It then (1) uses the reference evaluator to compute expected outputs, and (2) generates a C driver that declares the compiled function, calls it on the same inputs, and checks that the absolute difference between the compiled result and the reference value is below a fixed tolerance (e.g., 1e-9). The script compiles the LLVM IR to an object file, compiles the C driver, links them into a test binary, and runs it. Each function is reported as OK, parse_error, runtime_error (e.g., division by zero in the reference semantics), or unsupported (e.g., recursion). This setup gives me an automated way to test both the language front-end and the LLVM backend on hand-written examples and on real equations extracted from the arXiv and latex-formulas corpora.

# Evaluation
I evaluate LEL as a compiler rather than just a parser: for every accepted equation I generate LLVM IR, compile it with llc/clang, and link it against an auto-generated C test driver that calls the function on a small grid of inputs and checks the result against a Python reference evaluator over the same AST (within a fixed floating-point tolerance). 
I evaluated on two aspests. First, the performance (speed). 

And second, the correctness.
I run this pipeline on three corpora: (1) a tiny hand-written suite of 24 LEL functions, where all 22 non-recursive functions pass end-to-end; (2) a large sample from the Hugging Face latex-formulas dataset, where 360 MiTeX-normalized equations match the LEL shape and 119 of them pass the full pipeline; and (3) an arXiv robotics corpus, where 33 equations match the LEL subset and 9 compile and validate successfully. In all cases, failures are classified as parse errors, runtime errors in the reference semantics (e.g., division by zero or unknown symbols), or unsupported recursion.

In this section, I will first summary the overall result, and then dive deeper into the result on each dataset, and finally analyze the failure mode.
## Result overview
### Correstness
| Corpus                     | Raw equations | MiTeX OK → Typst | LEL-parsable | End-to-end OK |
|----------------------------|--------------:|-----------------:|-------------:|--------------:|
| Hand-written test suite    |            24 |               24 |           22 |           22  |
| `latex-formulas`  |       552,339 |          506,900 |          360 |          119  |
| arXiv|        42,896 |           29,356 |           33 |            9  |
### Speed
For functions that make it through parsing and reference evaluation, I also measure performance by timing the entire pipeline, from normalization and parsing through LLVM codegen, clang compilation, and execution of the test binary. On a M4 MacBook Pro laptop with 24 GB RAM running the Sequoia 15.6.1, processing 345 such functions takes 18.851 seconds total, for an average of about 54.6 ms per function, with a median of 0.1 ms and a maximum of ~512 ms. 

### On the tiny hand-written test
On the curated suite of 24 hand-written LEL functions covering arithmetic, powers, fractions, zero-argument functions, conditional expressions, and calls, our pipeline successfully parses, compiles, and validates 22/22 (100%) non-recursive functions. The two recursive examples (fact, accum) are intentionally treated as unsupported by our evaluation harness to avoid non-terminating test runs.

### On the Hugging face latex-formulas dataset 
On the large latex-formulas corpus (HuggingFace), 506,900 (about 92%) are successfully normalized by MiTeX into Typst math. 360 matched the LEL parser and are evaluated. Out of these, 119 (33%) functions passed, 226 failed with runtime_error, and 15 were marked unsupported due to explicit recursion (e.g., f(a) = f(alpha) + …, T(x) = T(-x)). The runtime_error cases are almost entirely due to my reference evaluator’s strict handling of free symbols: many equations involve undeclared constants, matrix/vector symbols, or built-in functions (e.g., x = f(x)), and the evaluator reports these as unknown variable rather than assuming numeric values. In contrast, the 119 successful cases are the clean subset where all identifiers can be interpreted purely as scalar parameters and local variables under my simple semantics. 

### On the ArXiv dataset
On the arXiv robotics corpus, MiTeX successfully produced normalized Typst math for 29,356 equations. Out of these, only 33 equations (~0.1%) matched the very restrictive LEL shape and were accepted by the LEL parser. When I ran the end-to-end evaluation pipeline on these 33 functions, 9 passed (27%) and 24 were reported as runtime_error. The failures were not due to codegen bugs but to the reference evaluator’s deliberately conservative treatment of free symbols: many equations use undeclared constants or matrix-valued symbols (e.g., Ax = b, R = det(M) - k(trace(M))^2), which the evaluator flags as "unknown variable" rather than silently treating them as numeric parameters. On real papers, the LEL compiler can successfully compile and run a small handful of clean scalar equations, while most extracted equations either fall outside the LEL subset or rely on ambient symbols (vectors, matrices, constants) that are not modeled in my current semantics. 

The passing rate on the arXiv corpus is slightly lower than on the Hugging Face dataset, likely because the arXiv equations are noisier (more custom macros, operators, and ambient symbols) and the evaluated sample from arXiv is smaller, so each failure has a larger impact on the overall percentage.


## Failure-mode analysis 
To understand why only a small fraction of real equations fall inside the LEL subset, I sampled a set of MiTeX-success but LEL-parse-fail equations from the HuggingFace corpus and manually categorized the errors.

| Failure category                         | Count (out of 50) |
|------------------------------------------|--------------------|
| No function header (`name(args) = expr`) | 18                 |
| Unsupported constructs (integral, matrix, etc.) | 22          |
| Identifier / normalization issues        | 10                 |


The main categories are:

1. **No LEL-style function header.**  
   Many equations are written as identities or operator equations rather than definitions, for example:
   - `ms + (ms+1)(s-1) = u(ms+1) - 1`
   - `\mathbb{E}(\mathcal{Z}^k_N(\beta)) \sim (...)^k \int ... \prod ...`
   - Operator forms such as `\mathcal{A} [u, w, \theta, q]^T = (...)`  

   These do not match LEL’s required `name(args) = expr` or `name = expr` header, so they are deliberately excluded.

2. **RHS uses constructs outside the LEL expression grammar.**  
   Even when there is a plausible "name–argument" pattern, the right-hand side often includes features that LEL does not support, such as:
   - Integrals and products (`\int`, `\prod`)
   - Matrices and arrays (`\begin{array}{cc} ... \end{array}`)
   - Special functions and macros as generated by MiTeX (e.g., `mitexsqrt[...]`, `mitexarray(...)`, `lr(...)`, `[q^d](...)|_{\epsilon = 0}`).  

   For example:
   - `\nu := \frac{\alpha\sqrt{1-\theta}}{\kappa} \min\{\lambda, \mu / \sqrt{1+\mu^2}\}`  
   - `\frac{d\varrho}{\varrho} = -(n+1)\frac{d\mathcal{R}}{\mathcal{R}}`  

   These are mathematically valid but lie outside the intentionally small expression language of LEL.

3. **Identifier and syntax normalization issues.**  
   MiTeX/Typst introduce additional structure into identifiers that does not fit LEL’s simple naming convention `[A-Za-z][A-Za-z0-9_]*`. Examples include:
   - Subscripts encoded as tokens like `Lambda \_g` instead of `Lambda_g`
   - Dotted names such as `phi.alt(...)`  

   For instance, the equation `\Lambda_g = AQ - D` becomes `Lambda \_g = AQ-D` in Typst, which fails LEL’s function-header regex. Similarly, names like `phi.alt` are not recognized as valid identifiers.

Overall, the failures are dominated by front-end issues (LaTeX constructs and MiTeX normalization) rather than bugs in the core compiler. Handling all the corner cases of real-world LaTeX—custom macros, operators, array/matrix syntax, and Typst’s own normalization quirks is an extremely tedious engineering problem. Fully solving this would require a substantial amount of additional front-end work and careful case-by-case normalization, which is well beyond the scope of a single course project.

# Hardest part
# GenAI usage
I used ChatGPT and Copilot throughout this project. 
I used GPT to:
- Brainstorm and outline what are the steps toward finishing this project
- Layoput a skeleton code
- Debug when there are issues
- Write scripts to do plotting and timing
- Refine the writing of the writeup
- Write the readme for the code repo
  
I used Copilot to:
- Completet the code with the inline plugin
- Debug when there are issues.