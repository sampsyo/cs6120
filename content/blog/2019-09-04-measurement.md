+++
title = "Measuring Computer Systems is Almost Certainly Harder Than You Think"
extra.author = "Adrian Sampson"
+++
Mytkowicz et al.'s ["Producing Wrong Data Without Doing Anything Obviously Wrong!"][paper] in ASPLOS 2009 is one of those papers that contains its own best summary.
Right there in Section 2, it says:

> Computer systems are sensitive: an insignificant and seemingly irrelevant change can dramatically affect the performance of the system.

The implication is profound:
even if you think reliably measuring computer systems is pretty hard,
it's probably harder than you think.

[paper]: https://dl.acm.org/citation.cfm?id=1508275


A Case Study in Measurement
---------------------------

In its most headline-worthy outcome,
this paper identifies two particularly irrelevant-seeming changes that can completely ruin a realistic experiment.
As a case study,
the authors set up a reasonable question that you can imagine wanting to answer empirically:
does [gcc][]'s `-O3` optimization level *really* offer any improvement over `-O2`?

It should be easy to answer this question: you just need to compile some programs at both optimization levels and measure how long they take to run.
But that obvious a sweeping assumption: that your handful measurements on those programs are representative of the *general concept* of compiling with different optimization levels.

Of course it's not, you might say—you can't possibly compile *all* the programs in the world, for example, so your choice of programs might certainly influence the results you see.
The best you can do is pick some standard, diverse, representative benchmarks and make the case that they're reasonably representative.
Similarly, while a truly robust finding would require measuring *all the computers in the world* that someone might ever run optimized binaries on, that's clearly infeasible—so you can do your job as a scrupulous scientist by measuring a few popular machines and hoping for the best.

The problem this paper identifies is that even a carefully designed experiment can go horribly wrong.
While benchmark and platform choice are clearly important factors, the paper finds factors that *do not seem important* can be just as critical.
When setting up this `-O2` vs. `-O3` experiment, for example, would you have guessed that the order of arguments in the linking command matters—that is, these two commands might compile binaries that run at different speeds?

    $ gcc foo.o bar.o -o bin
    $ gcc bar.o foo.o -o bin

I certainly would never have guessed that this detail could possibly matter.
The paper shows that it does, and at least for this case, it's every bit as important as the more obvious experimental design factors such as the choice of benchmark or the target CPU.

Specifically, the link order you choose can change your conclusion about whether `-O3` matters.
In the paper's experiments, certain "lucky" linking orders make it seem like `-O3` performs 8% better than `-O2`, while other "unlucky" linking order can perform 7% *worse*.
So if you ran this hypothetical experiment without trying multiple linking orders—and who would, honestly?—you could never know that your seemingly-solid results depend rigidly on a factor that has nothing to do with optimization levels at all.

I find the second confounding factor even more shocking.
Assuming the `bin` program does not ever read the `FOO` environment variable, would you expect these commands to run at different speeds?

    $ FOO=BAR ./bin
    $ FOO=BAAAAAAAAAAAAAAAAAR ./bin

This paper finds that they can.
Much like linking order, changing the size of Unix environment variables can cause large swings in execution time.
And these changes are *also* large enough to change the conclusion of the hypothetical `-O2` vs. `-O3` experiment.
Because usernames are part of the environment, this finding means that people with longer names might be more or less likely to observe benefits from compiler optimizations.

The paper does a thorough investigation into *why* these strange performance effects arise, and they're not all that surprising once you think of them:
environment variables can shift the starting stack address, for example, which can move data across cache lines, and CPUs care a lot about cache lines.
But these explanations are evidence in service a larger point:
computer systems are complex enough that their behavior can certainly surprise you.

[gcc]: https://gcc.gnu.org


A Model Evaluation
------------------

Aside from its actual research conclusions, the "Producing Wrong Data!" paper serves a second useful purpose:
as a model of a solid empirical evaluation.
No evaluation is perfect, but this one does a lot of things right.
If you hold the [SIGPLAN empirical evaluation guidelines][eeg] in one hand and this paper in the other, you can find lots of common ground.
Here are some standard pieces of evaluation advice that this paper exemplifies:

* *Use a complete, standard benchmark suite—and justify your choice.*
  This paper uses the most standard of standard suites, [SPEC][].
  It's careful to explain that it only uses the C benchmarks because Java compilers don't have `-O2` and `-O3` optimization levels.
* *Collect lots of data, even if takes a while.*
  This evaluation required 5,940 executions each of 12 benchmarks.
  For the slowest benchmark, collecting data took 12 days.
  A few CPU-weeks or even CPU-months can definitely be worth it if they give you solid results.
* *Thoroughly explore the design space.*
  The evaluation is not satisfied with just measuring one compiler on one system—the authors augment their main gcc results with Intel's [icc][], a second CPU, and even an architectural simulator.
* *Plot error bars.*
  It's good to measure the average of multiple runs in any experiment, and it's even better to report the standard error of the mean.
  All of the scatter and bar charts in this paper have error bars.
* *Report entire distributions.*
  Sometimes, error bars are not enough—it's even more useful to give a complete depiction of the data distribution.
  There are many ways to do this, but two good visualizations are violin plots and histograms.
  This paper uses both.
* *Include details about your experimental setup.*
  In any paper with an empirical evaluation, you have to include a complete description of the system you measured.
  Table 2 in this paper gives a complete breakdown of the platforms in these experiments.
* *Always explain your axes.*
  This paper contains several sentences that sound like "A point *(x,y)* says that when the UNIX environment size is *x* bytes, the execution time is *y* cycles on a Core 2 workstation."
  This description leaves absolutely no ambiguity about what the plot is telling us.
  It's easy to leave these descriptions off and make your data needlessly hard to interpret.
* *Dig deep to explain mysterious phenomena.*
  If you find something weird in your experimental results, it can be tempting to blame amorphous "measurement error" and leave it alone.
  Resist this temptation—look more closely and run more experiments to nail down exactly what's happening.
  This paper goes to heroic lengths to understand the microarchitectural reasons behind the outliers it found.

[eeg]: https://www.sigplan.org/Resources/EmpiricalEvaluation/
[spec]: https://www.spec.org
[icc]: https://software.intel.com/en-us/c-compilers


Lessons We're Still Learning
----------------------------

This paper is from 2009.
I last read it a long time ago, and upon re-reading it, I was saddened but not surprised to discover how many of its lessons still feel under-appreciated today.

Underlying the primary message about sensitivity is a secondary call to action:
computer systems researchers need to get serious about statistics.
Long ago, it's possible to imagine that computers were simple enough that experts could reasonably predict how they would behave.
Those days are gone.
While we should control as many factors as we can, we need to treat computer systems as unknowable, potentially hostile environments that actively try to spoil our experiments.
And one critical tool for rationalizing a complex, uncontrollable environment is good statistical analysis.
"Real" sciences like biology and chemistry seem to be more comfortable with this concept, but computer science lacks a strong tradition of rigorous statistics.

The best example of statistical thinking in this paper is embodied in the violin plots that show distributions of running times.
The key insight here is that it is *possible* to think of the different system treatments in these experiments as probability distributions.
You can imagine selecting a linking order uniformly at random from all the possible orders, for example, and sampling the execution time as a random variable.
It's obvious in retrospect, but it wasn't *necessary* for the authors to treat this data as random—after all, nobody *really* uses a random number generator to choose their linking order.
But viewing the data through a statistical lens helps give a complete picture of the confounding factor's influence.

As we read more papers throughout [6120][cs6120], let's keep the statistical attitude from Mytkowicz et al. in mind.
Let's keep a healthy skepticism about spurious findings and maintain a high standard for statistical sophistication.

[cs6120]: @/_index.md
