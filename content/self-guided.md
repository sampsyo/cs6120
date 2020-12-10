+++
title = "The Self-Guided Course"
template = "sg.html"
[extra]
hide = true
[extra.readings]
"1" = ["wrongdata", "eeg"]
"5" = ["ball-larus"]
"6" = ["alive"]
"9" = ["tbaa"]
"10" = ["ugc", "consgc"]
"11" = ["self", "tracemonkey"]
"12" = ["super", "chlorophyll"]
"13" = ["notlib", "slp", "dpj", "compcert"]
+++
# CS 6120: Advanced Compilers: The Self-Guided Online Course

CS 6120 is a PhD-level [Cornell CS][cs] course by [Adrian Sampson][adrian] on programming language implementation.
It covers universal compilers topics like intermediate representations, data flow, and “classic” optimizations as well as more research-flavored topics such as parallelization, just-in-time compilation, and garbage collection.
The work consists of reading papers and open-source hacking tasks, which use [LLVM][] and [an educational IR invented just for this class][bril].

This page lists the curriculum for following this course at the university of your imagination, for four imagination credits (ungraded).
There's a linear timeline of lessons interspersed with papers to read.
Each lesson has videos and written notes, and some have *implementation tasks* for you to complete.
Tasks are all open-ended, to one degree or another, and are meant to solidify your understanding of the abstract concepts by turning them into real code.
The order represents a suggested interleaving of video-watching and paper-reading.

Some differences with the “real” CS 6120 are that you can ignore the task deadlines and you can't participate in our discussion threads on Zulip.
Real 6120 also has an end-of-semester course project—in the self-guided version, your end-of-semester assignment is to change the world through the magic of compilers.

The instructor is a video production neophyte, so please excuse the production values, especially in the early lessons.
CS 6120 is [open source and on GitHub][gh], so please file bugs if you find problems.

[gh]: https://github.com/sampsyo/cs6120
[cs]: https://www.cs.cornell.edu/
[adrian]: https://www.cs.cornell.edu/~asampson/
[bril]: https://capra.cs.cornell.edu/bril/
[llvm]: https://llvm.org/
