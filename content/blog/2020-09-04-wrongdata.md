+++
title = "Taking Measurement Seriously"
[extra]
bio = """
[Adrian Sampson](https://www.cs.cornell.edu/~asampson/) is an assistant professor and the instructor for 6120. He harbors equal parts love and loathing for compilers.
"""
[[extra.authors]]
name = "Adrian Sampson"
link = "https://www.cs.cornell.edu/~asampson/"
+++
The ASPLOS 2009 paper [“Producing Wrong Data Without Doing Anything Obviously Wrong!”][paper] by [Mytkowicz][toddm] et al. is a modern classic and one of my favorite papers to recommend.
Instead of selling it to you anew, I’m going to refer you to [a blog post I wrote about it last year][2019post].
It’s still good—I swear!

After I wrote that blog post, someone referred me to a kind of kindred-spirit article: [John Ousterhout][ousterhout]’s 2018 CACM screed, [“Always Measure One Level Deeper”][ousterhout-cacm].
I recommend it if you want a perspective that takes the terrifying realizations from Mytkowicz et al. and turns them into concrete advice.

The common theme I want to take away for the CS 6120 context is that, in language implementation, *we must take measurement seriously*.
Compilers interact unforgivingly with the real world: not only with OS and hardware artifacts that skew performance unpredictably, but also with the wild deviations between language “specifications” and how actual people write code, with inconvenient facts of life like error messages, and the long tail of weird things that can happen when code you generate links with code you didn’t.
All these things mean that it is never safe to assume that something works—you need to demonstrate it empirically, and you need to assume that the universe is trying to trick you into reaching the wrong conclusion.

Here are some questions to get the discussion started:

* Humans invented computers. Why, philosophically, are we so bad at understanding their performance?
* Have you ever run into a Mytkowicz-like “dark matter” phenomenon when doing an empirical measurement?
* What’s your favorite of the [SIGPLAN empirical evaluation guidelines][eeg]? Which ones do you disagree with?
* Do you have any hilarious examples of violations of those guidelines?
* All evaluations are flawed. How do you know when it’s time to stop and say yours is good enough?

[ousterhout]: https://web.stanford.edu/~ouster/
[ousterhout-cacm]: https://cacm.acm.org/magazines/2018/7/229031-always-measure-one-level-deeper/fulltext
[2019post]: https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/measurement/
[paper]: https://dl.acm.org/citation.cfm?id=1508275
[toddm]: https://www.microsoft.com/en-us/research/people/toddm/
[eeg]: https://www.sigplan.org/Resources/EmpiricalEvaluation/
