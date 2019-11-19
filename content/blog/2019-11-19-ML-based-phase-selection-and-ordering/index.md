+++
title = "ML Based Phase Selection and Ordering"
extra.author = "Horace He & Qian Huang"
extra.latex = true
extra.bio = """
    Horace He is a senior undergraduate studying CS & Math. He's interested
    in the intersection of machine learning and compilers. This past summer,
    he worked on PyTorch at Facebook, as well as automatically scheduling
    Halide programs.

    Qian Huang is a junior undergraduate studying CS & Math. She's mostly
    interested in machine learning and algorithms, and was partially roped
    into the class by Horace. She's curious about the compiler optimizations
    people perform in practice.
"""
+++

Our goal was to 
# Background


# Design Overview

## Optimizations

To set up the phase ordering task, we selected and implemented several optimization passes:

- Dead Code Elimination (DCE) that we just used the given implementation directly.

- For data flow analysis based optimization passes, we implemented Copy Propagation and Constant Folding, as defined in the lecture. Each pass will gather data flow analysis information first then modify the instructions in place.

- For control flow graph optimizations, we implemented Branch Removal, Unreachable Code Elimination, CFG cleaning and  Tail Merging. We refer to this [slide](http://user.it.uu.se/~kostis/Teaching/KT2-10/Slides/ControlFlowOpts.pdf) for more details.
  
    + Branch Removal
  
        If we can decide the guard value to be true or false, then we can eliminate one of the two branches accordingly. 
    
    + Unreachable Code Elimination(UCE)

        The basic blocks unreachable from the entry block will be removed

    + CFG cleaning

        We can simplify CFG by several transformations that eliminate useless edges and combine some basic blocks. Specifically, we can 1) replace a br instruciton with two identical destination to a jmp instruciton; 2) merging empty blocks; 3) merging two blocks with only one edge in between under some cases. 

    +   Tail Merging
        
        If the predecesssor or successors share a same section of code in the end or begining correspondingly, we can change the jump to reuse the same segment of code.

In general, the optimization we selected are tailored for the simple structure of Bril, which has a lot of contant calculation and branches that do not allow falling through. Passes like DCE and UCE will likely to remove more codes, but other passes might or might not generate new opportunities for DCE and UCE. Thus we need to order passes properly to obtain best performance, especially under the case where we can only apply limited number of passes.



## Phase Ordering Analysis


## Phase Ordering Heuristics


# Evaluation 

## Random Program Generator 

## Performance

