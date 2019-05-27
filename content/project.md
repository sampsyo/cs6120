+++
title = "Projects"
+++
- They are *all* open ended.
    - We provide a list of possibilities; you can pick one or make up your own. You can do things that we cover in class or go far beyond and make up something totally new.
    - Commit to exactly what it will be up front.
    - In just one sentence, summarizing your intent for each of the three parts: what you're doing, how you'll do it, and how you'll measure success.
- The project itself consists of:
    - Designing exactly what you will do.
    - Implementing it (in code).
        - This must be open source.
        - You're allowed/encouraged to use your own or others' implementations. Just not to do exactly the same thing. If you want to do the same thing, that's OK, but you can't use existing code.
    - Measuring success. You'll need to consider:
        - Benchmarks/input programs.
        - Checking correctness.
        - Measuring performance.
        - Presentation of data.
- Writeup.
    - Post on on our (git-based) blog.
    - Answer these questions:
        - What was the goal?
        - What did you do? (Design & implementation.)
        - What was the hardest part?
        - Were you successful? (OK if no! Just have to be rigorous.)
- Rotating in-class *design reviews* following the same structure.
    - Everyone will do N of these. There's some flexibility in which project you do the design review for.
    - Everyone submits (grade-free) paragraphs on the design review from the day. (Do this in class.)
- Grading is based on:
    - ambition
    - clarity
    - rigor

---

Projects:

1. Warm up. Build something simple that uses the Bril compiler infrastructure.
    - Basic
        - CFG visualizer (using dot)
        - constant propagation optimization
        - dynamic profiling
    - Advanced
        - a Bril extension: higher-order functions; a new datatype; C ABI FFI
        - wasm backend
        - a new frontend for your favorite language
        - a fast interpreter (binary representation, direct threading)
2. Classic optimization for Bril.
    - into & out of SSA
    - global value numbering
    - a different common subexpression elimination (CSE) optimization
    - generic data flow solver, e.g., Kildall’s algorithm
    - pointer analysis
3. Do something with LLVM.
    - another classic optimization, this time "for real"?
4. (and thereafter)... advanced?
