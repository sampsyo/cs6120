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

(Zulip is an open-source Slack-like chat tool with a few unique features that make it a good fit for class discussion. Thanks to [the Zulip folks][zulip-co] for donating a plan for 6120!)

[zulip]: https://cs6120.zulipchat.com
[zulip-signup]: https://www.cs.cornell.edu/courses/cs6120/2022sp/private/zulip.html
[zulip-co]: https://zulipchat.com


## Class Sessions

The scheduled class sessions for 6120 have a funky flipped format this year.
Everything happens on Zoom (please join [Zulip][] for the link).
Check the [schedule][] for each day of class—it will fall into one of two categories:

* *"Lesson" days:*
  Before class, watch the (required) video and, if you like, do the optional associated reading.
  While or after watching the video, ask questions in the associated GitHub Discussions thread or on Zulip.
  The class will be a quiet hacking session and an opportunity to ask questions and work together with the rest of the class on [implementation tasks](#implementation-tasks).
  To participate virtually, hang out on Zulip and ask questions while you work.
* *"Discussion" days:*
  Before class, do the (required) reading and answer the (required) discussion questions on the associated GitHub Discussions thread.
  We will discuss the reading synchronously (on Zoom or in person).
  After the discussion wraps up, there may be time for more hanging around and hacking.


## The Work

There are four kinds of things that make up the coursework for CS 6120.

* *Implementation tasks:*
  Discrete bits of compiler implementation work to deepen your understanding of the topics in the [lessons][].
* *Paper reading:*
  We'll read research papers, and everybody will participate in asynchronous (pre-class, online) discussions and synchronous (in-class) discussions.
* *Leading paper discussions:*
  For a subset of the aforementioned research papers, you'll be responsible for leading the online and synchronous discussion and synthesizing a [blog][] post.
* *Course project:*
  There is an open-ended course implementation project that culminates in a blog post.

[lessons]: @/lesson/_index.md
[blog]: @/blog/_index.md

### Implementation Tasks

To reinforce the specific compiler techniques we learn about in class, you will work on implementing them on your own.
The usual pattern is that we will come up with the high-level idea and the pseudocode in lessons; your job is to turn it into real, working code and collect empirical evidence that it's working.
Going through the complete implementation will teach you about realities that you cannot get by thinking at a high level.

You can work individually or in groups of 2–3 students.
When you finish an implementation, do this:

* I recommend (but do not require) that you put all your code online in an open-source source code repository, e.g., on GitHub.
* Include a short README (just a paragraph is fine) describing what you did and how you know your implementation works.
* Submit the assignment on [CMS][]. Just submit a text file with a URL to your open-source implementation if it's available. If you for some reason don't want to open-source your code, you can instead upload the code itself.
* Write a brief post in the lesson's associated GitHub Discussions thread covering the following topics (just a paragraph or so is fine):
    * Summarize what you did.
    * Explain how you know your implementation works---how did you test it? Did you use any qualitative or quantitative experiments?
    * What was the hardest part of the task? How did you solve this problem?

Do all your own work on implementation tasks.
I have sample implementations for many of the tasks in the 6120 GitHub repository; you may not use this code.
Past 6120 students have open-sourced their implementations; you may not use this either.
I recommend not looking at my or others' implementations at all while working on tasks, so you actually learn how to do it---but it's not hiding if you absolutely need it.
You are an adult, presumably, and can control your own impulse to skip the hard work.

### Paper Reading & Discussion

Another part of 6120 is reading and discussing research papers.
Every time we read a paper (see the [schedule][]), everybody participates in the discussion in two ways:
through [GitHub Discussions][gh-disc] threads before class,
and then synchronously in class.
For every paper, there will be a Discussions topic; post at least one message there with your thoughts on the paper before the class period with the discussion.
Your comment need not be long at all; just a couple of sentences is fine.
And it need not be a top-level post; I encourage you to respond to other people's thoughts on the thread.

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
* Collect a list questions for the in-class discussion.
  You can write your own or steal the best from the online discussion.

*On* the discussion day:

* Moderate a short discussion.
  There is no presentation component to the discussion—just set up enough background to get to the discussion questions and help move the conversation along.

Due *one week after* the discussion day:

* Write a post about the paper for our [course blog][blog].
  The post should contain:

     * Any background that you think the audience needs to understand the paper.
     * A detailed summary of the main contributions in the work.
     * Critical thinking about the merits and shortcomings of the work.
     * Discussion of the work's role in history and its connections to the current computing landscape.

    Feel free to incorporate the best ideas you heard during the online and in-class discussion.
    You can represent your own opinions, ones from the class as a whole, or both.

    If you need inspiration for the style of post to write, check out [last year's blog][blog2020]. But probably avoid reading posts about your paper, if they exist!
* Optionally, you can record a video to go along with your blog post that people should watch to get the discussion started.
* Publish the post to the [course GitHub repository][gh] by opening a pull request.
  The repository README has instructions.
* When your PR is open, announce it on the appropriate GitHub Discussions thread to let other people take a look.

[schedule]: @/schedule.md
[blog2020]: https://www.cs.cornell.edu/courses/cs6120/2020fa/blog/
[gh]: https://github.com/sampsyo/cs6120
[gh-disc]: https://github.com/sampsyo/cs6120/discussions
[gh-disc-reading]: https://github.com/sampsyo/cs6120/discussions/categories/reading

### Project

At the end of the course, you'll do a language implementation research project.
This is an open-ended and open-source project that can be on any topic that you can construe as being about compiler hacking.
The final product is an experience report on the [course blog][blog] where you rigorously evaluate the success of your implementation.

You can work individually or in groups of 2–3 people.

#### Proposal

The first deadline is the project proposal.
[Open a GitHub issue][proposal] answering these three questions, which are a sort of abbreviated form of the [Heilmeier catechism][hc]:

* What will you do?
* How will you do it?
* How will you empirically measure success?

You should also list the GitHub usernames of everyone in the group.
After you send the PR, submit its URL to the "Project Proposal" assignment on [CMS][].

The instructor will have feedback on how to approach your project.

[hc]: https://www.darpa.mil/work-with-us/heilmeier-catechism
[proposal]: https://github.com/sampsyo/cs6120/issues/new?labels=proposal&template=project-proposal.md&title=Project+%5BNUMBER%5D+Proposal%3A+%5BTITLE%5D

#### Implementation

The main phase, of course, is implementing the thing you said you would implement.
I recommend you keep a “lab notebook” to log your thoughts, attempts, and frustrations—this will come in handy for the report you'll write about the project.

I strongly recommend that you develop your code as an open-source project.
Use a publicly-visible version control repository on a host like GitHub, and include an [open source license][osi].
When you create your repository, comment on your proposal GitHub issue with a link.
(If you have a specific objection to open-sourcing your code, that's OK—include a description of how you'll share your code privately with me.)

[osi]: https://opensource.org/licenses

#### Evaluation

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

#### Experience Report

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

I will grade the four categories of coursework based on a [Michelin star][michelin] system:
you should shoot for one star, indicating excellent implementation work, insightful participation, a thoughtful blog post, or a spectacularly successful project.
Merely acceptable work will receive no stars, and extraordinary work can receive multiple stars.
On [CMS][], all assignments are out of one point, indicating the number of stars.

Consistently earning a Michelin star will get you an A for the course.
Occasional multi-star work yields an A+, and missing stars leads to incrementally lower grades.

[michelin]: https://en.wikipedia.org/wiki/Michelin_Guide
[cms]: https://cmsx.cs.cornell.edu/


## Policies

### Non-PhD Enrollment

CS 6120 is for PhD students at Cornell.
If that doesn’t describe you, you can still apply to take the course.
Write short answers to these two questions:

1. What is CS 6120 about?
2. Why do you want to take the course?

Submit your responses as a one-page PDF to Adrian via [Zulip][].

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
