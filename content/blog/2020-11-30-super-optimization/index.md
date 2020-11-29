+++
title = "Super Optimization"
[extra]
bio = """
  Socrates Wong is a secound year PhD student at Cornell who intreset lies in Computer Archicture.
"""
latex = true
[[extra.authors]]
name = "Socrates Wong"
+++

## Background
Most complier optimization (peephole, dead code elimination, constant propagation, loop invariant code motion, common sub expression elimination) focus on transforming the program in a systematic manner, and the step taken by the optimizer are backed by formal rigorous proofs that are both sound and complete.  Superoptimizer is a new radical way of preforming optimization.  Instead of taking the traditional approach optimizers of transforming the input program, superoptimizer converts the problem in to a search space and attempts to find the shortest program that is able to provide the equivalent output.
The paper we are discussing this week is ["Superoptimizer"][superopt] – A look at the Smallest Program” (Henry), this paper explains the purpose of superoptimizer, which converts an optimizing problem to a search space, how to reduce the search space and comparison with other common optimization techniques that are common at the publication of the paper. 


## Super Optimization
Super Optimization is finding the shortest program by exhaustive search inside a finite search space of the instruction set architecture (ISA).  Compare to other optimization techniques that focus on transformation and match, super optimization takes it to a whole different level.  Most optimizations techniques focus on the potential performance benefit on utilizing a transformation and ensuring correctness of the output by having proofs that the transformation does not affect the equivalence of the original program.  In modern compliers, this can ensure that the result is unchanged, but does not guarantee optimality of the output.  To address the need for the most optimal output, super optimization searches the entire sample spaces of the instruction set architecture for programs by generating all ombinations of eah instruction with various lengths, starting with the shortestlength.  

Then super optimization then preform a test of the generated program determine if the generated program is equivalent with the orgnial program and if they are the same, then generated programwill be the optimal porgram for the ISA.  Although this soulds simple, verifing the generated program is a non trival task.  One of the sloution proposed in the paper was boolean test.  However as the complexity of some instructions have expressio of the order of 2^31 minterms (Massalin), it has caused a constriant on the number of instrcitios that can be generated and verified in a reasonable time to be limited to 3 instructions at the time of publication.  Therefore although the boolean test is intuitive, and correct a alternative test will be needed in order to test programs wiht larger amount of instructions.  

The paper also has proposed a probailistict execution test in which the superoptimizer picks a test vectr and see if the output results are the same between the test and generated program.  Although this test is not compprehansive, the author claims that there was never a case where probailistict execution test passes while boolean test fails.  The time it takes for exhaustive search can be futher reduced by ulitization of pruing on the generated sequences that are equivalent to shorter sequences.  

## Results

One of the most intresting part of this paper is the results.  Although compare to other papers where they have a long compression of performance, superoptimizer reported it's result by comparing the execution of the input porgram and assembly.  For example, given the following signum program: 
<img src="superin.png" style="width: 30%;">

Getting signum program to be immplemented in less then 9 assembly statments is already hard.  But superoptimizer managed to convert the program into a series of adds, subtracts, negation without any branches.  Resulting in the following results:
<img src="superout.png" style="width: 30%;">
By removing branches and the number of instructions, this immplemation has provide huge benfits to morden out of order processors (remove the need of branch predictions) and in both number of execution cycles.    Other results are similar interesting and that demestrated clearly the different then what a regular complier would generate.  

## Limitations, Merits and Shortcomings

One of the biggest limitaiton of super optimization would have been complier time cost and the benfits taht can be obtained at run time.  Although super optimization can ensure optimality, it does not nessciarity ensure a preformance bouns over exisitng solutions. That being said, this paper has a great impact in both llvm and gcc communities.  Despite the super optimization is not being actively used in code compliation that they each have their own super optimization immplemention, which is partly inspirted by this paper. Although this paper has strong merits on the grounds of creating optimal code in applications and usage, it is not very pratical in most contempary compliers.  


## Connections to Modern Works

Super optimizations desipte having good results has not be very wide adopted in for compliation compliers framework since this paper has been writen.  This concept has been reintroduced as papers such as [Automatic generation of peephole superoptimizers (Bansal, Aiken 2006)] [autopep]  Although the usage was wastly different from the orginal form of super optimization, this cencept managed to endure the test of time and leave its make in peepholde generation.  

## Discussion Questions

- The landscape in the discipline of search has evolved greatly since the publication of the paper in 1987,  especially in aspects such as algorithms, and computational capacity of modern hardware.  To what extent does this affects superoptimizier, and what changes would see if this paper is written now?
- In the discipline of search in computer science, there exist a type of search called heuristics.  To what extent do you think it is acceptable to this topic?
- •	Although the two most common complier infrastructure (LLVM and GCC) have a superoptimizier implemented for their infrastructure, it is not currently used in regular complier sequence.  To what extent is it due to the performance at compilation time, the maturity of super optimizer, the lack of interest of finding the most optimal program, or some other reason that is blocking it from being adopted as a standard compilation tool?

[superopt]: https://courses.cs.washington.edu/courses/cse501/15sp/papers/massalin.pdf
[autopep]:https://theory.stanford.edu/~aiken/publications/papers/asplos06.pdf
