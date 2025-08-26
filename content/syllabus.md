+++
title = "Syllabus"
[extra]
hide = false
+++
# CS 6120: Advanced Compilers Syllabus

This is a research-oriented PhD course about programming language implementation.
The class focuses on hacking, reading papers, and writing [blog][] posts.


## Communication

Please [sign up][zulip-signup] for the [course Zulip][zulip].
All course communication will happen there (not on email).
You will need [this magical, Cornell-authenticated invitation link][zulip-signup] to register for Zulip.

(Zulip is an open-source Slack-like chat tool with a few unique features that make it a good fit for class discussion. Thanks to [the Zulip folks][zulip-co] for donating a plan for 6120!)

[zulip]: https://cs6120.zulipchat.com
[zulip-signup]: https://www.cs.cornell.edu/courses/cs6120/2025fa/private/zulip.html
[zulip-co]: https://zulipchat.com


## Class Sessions

Check the [schedule][] for each day of class—it will fall into one of two categories:

* *"Lesson" days:*
  Before class, optionally watch the video and/or do any associated reading.
  The video covers approximately the same stuff as we'll talk about in class, so don't sweat it.
  The class may also include a quiet hacking session and an opportunity to ask questions and work together with the rest of the class on [implementation tasks](#tasks).
* *"Discussion" days:*
  Before class, do the (required) reading and answer the (required) discussion questions on the associated GitHub Discussions thread.
  We will discuss the reading synchronously.
  After the discussion wraps up, there may be time for more hanging around and hacking.


## The Work

There are four kinds of things that make up the coursework for CS 6120.

* [*Implementation tasks:*](#tasks)
  Discrete bits of compiler implementation work to deepen your understanding of the topics in the [lessons][].
* [*Paper reading and discussion:*](#papers)
  We'll read research papers, and everybody will participate in asynchronous (pre-class, online) discussions and synchronous (in-class) discussions.
* *Leading paper discussions:*
  For a subset of the aforementioned research papers, you'll be responsible for leading the online and synchronous discussion and synthesizing a [blog][] post.
* [*Course project:*](#project)
  There is an open-ended course implementation project that culminates in a blog post.

If you are considering whether to enroll in this class,
please think carefully about whether all of these forms of work appeal to you.
Some examples of work styles that some people don't like include writing blog posts for a general audience and organized
small-group discussions.

[lessons]: @/lesson/_index.md
[blog]: @/blog/_index.md


## Implementation Tasks {#tasks}

To reinforce the specific compiler techniques we learn about in class, you will work on implementing them on your own.
The usual pattern is that we will come up with the high-level idea and the pseudocode in lessons; your job is to turn it into real, working code and collect empirical evidence that it's working.
Going through the complete implementation will teach you about realities that you cannot get by thinking at a high level.

*Testing* your implementation is a critical component.
Your job is to make a convincing case that your implementation does what you intended it to do: for example, an optimization is supposed to make programs faster, most of the time, and never break any programs.
A formal proof of these properties is probably out of scope, so you will need to find some other way to gather evidence.
In particular, *don't (just) use existing test cases you find in the Bril git repository,* which are never thorough enough.
One good tactic can be to use [all the benchmarks in the repo][bench], however.

You can work individually or in groups of 2–3 students.
When you finish an implementation, do this:

* I recommend (but do not require) that you put all your code online in an open-source source code repository, e.g., on GitHub.
* Submit the assignment on [CMS][]. Just submit a text file with a URL to your open-source implementation if it's available. If you for some reason don't want to open-source your code, you can instead upload the code itself.
* Write a brief post in the lesson's associated GitHub Discussions thread covering the following topics (just a paragraph or so is fine):
    * Summarize what you did.
    * Explain how you know your implementation works---how did you test it? Which test inputs did you use? Do you have any quantitative results to report?
    * What was the hardest part of the task? How did you solve this problem?
    * Do you think your work deserves a [Michelin star](#grading)?

Do all your own work (with your group) on implementation tasks.
I have sample implementations for many of the tasks in the 6120 GitHub repository; you may not use this code.
Past 6120 students have open-sourced their implementations; you may not use this either.
I recommend not looking at my or others' implementations at all while working on tasks, so you actually learn how to do it---but it's not hiding if you absolutely need it.
You are an adult, presumably, and can control your own impulse to skip the hard work.

If you're willing, I'm seeking volunteers to track the time they spend on each task.
You would use something like [Toggl][] to set a separate timer for each assignment, and tell me the total number of minutes you spent on each one.
This will help me calibrate the time allotted for 6120 tasks in future offerings.

[bench]: https://capra.cs.cornell.edu/bril/tools/bench.html
[toggl]: https://toggl.com


## Paper Reading & Discussion {#papers}

Another part of 6120 is reading and discussing research papers.
Every time we read a paper (see the [schedule][]), everybody participates in the discussion in two ways:
through [GitHub Discussions][gh-disc] threads before class,
and then synchronously in class.
Someone will take on the role of the discussion leader for both components.

### Before Class

For every paper, the leader will create a GitHub Discussions topic; post at least one message there with your thoughts on the paper before the class period with the discussion.
You can either start your own top-level subthread or respond to someone else's.

Your comment must include both of these elements (and anything else you want to say):

* Some **critique** of the paper: your own thoughts that expand on some element in the paper. While "critique" may have negative connotations, this does not need to be negative: extrapolations that express appreciation, curiosity, or just technical extensions also count.
* At least one **discussion question** that you would like to address synchronously with the class. For examples of good and bad discussion questions, see [Lindsey Kuper's instructions for UCSC's CSE290S][290s-questions].

Comments are due **at 10pm on the day before the discussion** so the leader has a chance to look before class.

[290s-questions]: https://decomposition.al/CSE290S-2023-01/course-overview.html#how-to-write-discussion-questions

### In Class

Discussion days have three phases:

1. Discussion lead's overview and questions: 10 minutes.
2. Small-group discussion: 30 minutes.
3. Full-class discussion: 30 minutes.

The first phase is a short (slides-free) overview that distills the online discussion so far and poses the discussion questions we'll be using.

In the second phase, we'll divide into randomly assigned groups of 3--4 people.
Each group will do these things:

* Introduce yourselves.
* Talk about the selected discussion questions.
* Try to reach consensus on the group's answers.
* Prepare to present these answers to the full class.

In the third phase, the leader will moderate a full-class discussion on each of the questions.
Each small group will report their answers in 3 minutes or less, and then we'll try to reach a broader consensus on the questions.

### Leading the Discussion

For some subset of the papers, you will be the discussion leader!
Leaders have three extra responsibilities:
monitoring and replying to the asynchronous discussion,
moderating and guiding the in-class discussion,
and synthesizing some ideas into a blog post after the fact.

*At least a week before* the discussion day:

* Create a GitHub Discussions thread in the [Reading][gh-disc-reading] category for your topic.

*During the lead-up to* the discussion day:

* Watch the GitHub Discussions thread for your topic.
  Answer any questions you can, offer extra insight when you have it, and so on.
* Select the 2--3 questions you will use for the in-class discussion.

*On* the discussion day:

* Briefly present an overview of the asynchronous discussion.
  **This is not a presentation about the content of the paper, which everyone has already read.**
  Instead, your job is to distill key points from the asynchronous discussion and to list your selected discussion questions for the day.
  **Do not prepare slides, and do not use more than 10 minutes.**
  You can project the paper itself if you need visual aids, and display your questions using a text editor or something.
* Moderate the full-class discussion phase.

Due *one week after* the discussion day:

* Write a post about the paper for our [course blog][blog].
  The post should contain:

     * Any background that you think the audience needs to understand the paper.
     * A detailed summary of the main contributions in the work.
     * Critical thinking about the merits and shortcomings of the work.
     * Discussion of the work's role in history and its connections to the current computing landscape.

    Feel free to incorporate the best ideas you heard during the online and in-class discussion.
    You can represent your own opinions, ones from the class as a whole, or both.

    While the second item above entails recapping some of the paper, **please resist the temptation to let direct summary make up the bulk of your blog post**. Keep this kind of technical explanation under around a quarter of the length. (To make this work, you will need to prioritize breadth over depth in your summary, and you'll have to pick and choose specific contributions to highlight instead of hoping to convey the entire paper.)

    Focus most of your writing on your own commentary: context, criticism, and discussion. To emphasize this focus, consider choosing a title for your blog post that is not the title of the paper---instead, it should reflect the main point *you want to make* about the paper.

    If you need inspiration for the style of post to write, check out [last year's blog][blog2020]. But probably avoid reading posts about your paper, if they exist!
* Publish the post to the [course GitHub repository][gh] by opening a pull request.
  The repository README has instructions.
* When your PR is open, announce it on the appropriate GitHub Discussions thread to let other people take a look.

[schedule]: @/schedule.md
[blog2020]: https://www.cs.cornell.edu/courses/cs6120/2020fa/blog/
[gh]: https://github.com/sampsyo/cs6120
[gh-disc]: https://github.com/sampsyo/cs6120/discussions
[gh-disc-reading]: https://github.com/sampsyo/cs6120/discussions/categories/reading


## Project {#project}

At the end of the course, you'll do a language implementation research project.
This is an open-ended and open-source project that can be on any topic that you can construe as being about compiler hacking.
The final product is an experience report on the [course blog][blog] where you rigorously evaluate the success of your implementation.

You can work individually or in groups of 2–3 people.

### Proposal

The first deadline is the project proposal.
[Open a GitHub issue][proposal] answering these three questions, which are a sort of abbreviated form of the [Heilmeier catechism][hc]:

* What will you do?
* How will you do it?
* How will you empirically measure success?

You should also list the GitHub usernames of everyone in the group.
After you send the PR, submit its URL to the "Project Proposal" assignment on [CMS][].

I will give you feedback on how to approach your project.
I might also ask you to make changes or add details about your plan; please respond to these requests quickly.
Once we have a plan, *please consider that a contract:*
if you need to change the plan, let me know by commenting on your GitHub issue thread and we will work something out.
I will grade your project based on how well it fulfills the plan we agreed on.

[hc]: https://www.darpa.mil/work-with-us/heilmeier-catechism
[proposal]: https://github.com/sampsyo/cs6120/issues/new?labels=proposal&template=project-proposal.md&title=Project+%5BNUMBER%5D+Proposal%3A+%5BTITLE%5D

### Implementation

The main phase, of course, is implementing the thing you said you would implement.
I recommend you keep a “lab notebook” to log your thoughts, attempts, and frustrations—this will come in handy for the report you'll write about the project.

I strongly recommend that you develop your code as an open-source project.
Use a publicly-visible version control repository on a host like GitHub, and include an [open source license][osi].
When you create your repository, comment on your proposal GitHub issue with a link.
(If you have a specific objection to open-sourcing your code, that's OK—include a description of how you'll share your code privately with me.)

[osi]: https://opensource.org/licenses

### Evaluation

A major part of your project is an empirical evaluation.
To design your evaluation strategy, you will need to consider at least these things:

* Where will you get the input code you'll use in your evaluation?
* How will you check the correctness of your implementation?
  If you've implemented an optimization, for example, “correctness” means that the transformed programs behave the same way as the original programs.
* How will you measure the benefit (in performance, energy, complexity, etc.) of your implementation?
* How will you present the data you collect from your empirical evaluation?

Other questions may be relevant depending on the project you choose.
Consider the [SIGPLAN empirical evaluation guidelines][eeg] when you design your methodology.

[eeg]: https://www.sigplan.org/Resources/EmpiricalEvaluation/

### Experience Report

For the main project deadline, you will write up the project’s outcomes in the form of a post on the [course blog][blog].
Your writeup should answer these questions in excruciating, exhaustive detail:

* What was the goal?
* What did you do? (Include both the design and the implementation.)
* What were the hardest parts to get right?
* Were you successful? (Report rigorously on your empirical evaluation.)

As with paper discussions, you can optionally include a video to go along with your blog post.

To submit your report, open a pull request in [the course’s GitHub repository][gh] to add your post to the blog.
In your PR description, please include “closes #N” where N is the issue number for your proposal.
The repository README has instructions.

[gh]: https://github.com/sampsyo/cs6120


## Grading

All four categories of work in CS 6120 use a [Michelin star][michelin] grading system:
you should shoot for one star, indicating excellent implementation work, insightful participation, a thoughtful blog post, or a spectacularly successful project.
Merely acceptable work will receive no stars, and extraordinary work can receive multiple stars.
On [CMS][], all assignments are out of one point, indicating the number of stars.

Consistently earning a Michelin star will get you an A for the course.
Occasional multi-star work yields an A+, and missing stars leads to incrementally lower grades.

For the [implementation tasks](#tasks) only, you will perform a self-assessment, i.e., your submission must include your own opinion about whether you deserve a Michelin star.
I will take your opinion into account when assigning your grade.

[michelin]: https://en.wikipedia.org/wiki/Michelin_Guide
[cms]: https://cmsx.cs.cornell.edu/


## Policies

### Non-PhD/MS Enrollment {#enroll}

CS 6120 is for PhD & MS students.
If that doesn't describe you, you can still apply to add the course during the add/drop period.
Write short answers to these two questions:

1. What is CS 6120 about?
2. Why do you want to take the course?

Submit your responses as a one-page PDF to Adrian via [Zulip][].
Please do this after the first class session, so you have some context about the course, and before the second (i.e., by January 23, 2025).
If you miss that deadline, I would strongly prefer that you do not try to add the course.
I will approve your application via Zulip.
Then, in about 2 working days, the CIS registration office will send you a PIN to enroll.

If you have questions about the PIN system, you can [file a request with the CIS registration office][reg-ticket].
But they have asked me to convey that you should not file a ticket merely to ask, "Adrian has approved me; may I please have a PIN?"
I will send them a list of people who can enroll, and then there is nothing for you to do but wait.

[reg-ticket]: https://tdx.cornell.edu/TDClient/193/Portal/Home/

### Academic Integrity

Absolute integrity is expected of all Cornell students in every academic undertaking. The course staff will prosecute violations aggressively.

You are responsible for understanding these policies:

- <a href="http://cuinfo.cornell.edu/Academic/AIC.html">Cornell University Code of Academic Integrity</a>
- <a href="http://www.cs.cornell.edu/ugrad/CSMajor/index.htm#ai">Computer Science Department Code of Academic Integrity</a>

You can also read about the [protocol for prosecution of violations][aiproceedings].

[aiproceedings]: http://www.theuniversityfaculty.cornell.edu/AcadInteg/index.html

Everything you turn in must be ether 100% completely your own work or clearly attributed to someone else.
You may discuss your work with other students, look for help online, get writing feedback from the instructor, ask your friend to help debug something, or anything else that doesn't involve someone else doing your work for you.
You may not turn in any writing that you did not do yourself or misrepresent existing implementation work as your own.

The projects in this course are open source, and you may use existing code in your implementation—including code written by other students in this course. You must, however, make it clear which code is yours and which code is borrowed from where.

### Generative AI

The work you do for CS 6120 consists of writing code and natural language descriptions.
To some extent, the new crop of ["generative AI" (GAI)][gai] tools can do both of these things for you.
In this class, you can choose---for every assignment---between two options:

1. Avoid all GAI tools. Disable [GitHub Copilot][copilot] in your editor, do not ask chatbots any questions related to the assignment, etc. If you choose this option, you have nothing more to do.
2. Use GAI tools, and include a one-paragraph description of everything you used them for along with your writeup. This paragraph must:
    * Link to exactly which tools you used and describe how you used each of them, for which parts of the work.
    * Give at least one concrete example (e.g., generated code or Q&A output) that you think is particularly illustrative of the "help" you got from the tool.
    * Describe any times when the tool was *unhelpful*, especially if it was wrong in a particularly hilarious way.
    * Conclude with your current opinion about the strengths and weaknesses of the tools you used for real-world compiler implementation.

Remember that you can pick whether to use GAI tools *for every assignment*, so using them on one set of tasks doesn't mean you have to keep using them forever.

[gai]: https://teaching.cornell.edu/generative-artificial-intelligence/cu-committee-report-generative-artificial-intelligence-education#Recommendations
[copilot]: https://github.com/features/copilot/

### Respect in Class

Everyone—the instructor, TAs, and students—must be respectful of everyone else in this class. All communication, in class and online, will be held to a high standard for inclusiveness: it may never target individuals or groups for harassment, and it may not exclude specific groups. That includes everything from outright animosity to the subtle ways we phrase things and even our timing.

For example: do not talk over other people; don't use male pronouns when you mean to refer to people of all genders; avoid explicit language that has a chance of seeming inappropriate to other people; and don't let strong emotions get in the way of calm, scientific communication.

If any of the communication in this class doesn't meet these standards, please don't escalate it by responding in kind. Instead, contact the instructor as early as possible. If you don't feel comfortable discussing something directly with the instructor—for example, if the instructor is the problem—please contact the advising office or the department chair.

### Special Needs and Wellness

We provide accommodations for disabilities.
Students with disabilities can contact <a href="http://sds.cornell.edu">Student Disability Services</a> at
607-254-4545 or the instructor for a confidential discussion of their
individual needs.

If you experience personal or academic stress or need to talk to someone who can help, contact the instructor or:

- <a href="http://www.engineering.cornell.edu/student-services/academic-advising">Engineering Academic Advising</a> at 607-255-7414
- <a href="http://lsc.sas.cornell.edu">Learning Strategies Center</a> at 607-255-6310
- <a href="http://www.gannett.cornell.edu/LetsTalk">Let's Talk Drop-in Counseling</a> at Gannett at 607-255-5155
- <a href="http://ears.dos.cornell.edu">Empathy Assistance and Referral Service</a> at 607-255-EARS
