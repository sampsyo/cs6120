+++
title = "Projects"
+++
# Language Implementation Projects

There are several implementation projects in this course.
Here’s what you do for each one:

* Propose a project and submit a short description of your plan.
* Implement something, according to the project description.
* Release your code as open-source software.
* Measure your implementation empirically and draw rigorous conclusions. (Negative results are OK! But flimsy evaluation is not.)
* Write a report as a post on our [course blog][blog].
* Do an in-class design review to collect feedback from your peers.

[blog]: @/blog/_index.md


## Proposal

At the beginning of the project period, you will decide what to do for the project.
Each project has guidelines for in-scope projects and will include a list of ideas, but you can also pick your own.

To propose a project, [open a GitHub issue][proposal] answering these three questions, which are a sort of abbreviated form of the [Heilmeier catechism][hc]:

* What will you do?
* How will you do it?
* How will you empirically measure success?

The instructor may have feedback for your or just tacitly approve your idea.

[hc]: https://www.darpa.mil/work-with-us/heilmeier-catechism
[proposal]: https://github.com/sampsyo/cs6120/issues/new?labels=proposal&template=project-proposal.md&title=Project+%5BNUMBER%5D+Proposal%3A+%5BTITLE%5D


## Implementation

- The project itself consists of:
    - Designing exactly what you will do.
    - Implementing it (in code).
        - This must be open source.
        - You're allowed/encouraged to use your own or others' implementations. Just not to do exactly the same thing. If you want to do the same thing, that's OK, but you can't use existing code.


## Evaluation

- Measuring success. You'll need to consider:
    - Benchmarks/input programs.
    - Checking correctness.
    - Measuring performance.
    - Presentation of data.

## Report

- Writeup.
    - Post on on our (git-based) blog.
    - Answer these questions:
        - What was the goal?
        - What did you do? (Design & implementation.)
        - What was the hardest part?
        - Were you successful? (OK if no! Just have to be rigorous.)


## Design Review

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
