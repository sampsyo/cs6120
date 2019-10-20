+++
title = "Out of the Loop!"
[extra]
latex = true
bio = """
[Rolph Recto](https://twitter.com/rolphrecto) is a third-year graduate student studying
the intersection of programming languages, security, and distributed systems. Likes climbing, scifi, and halo-halo.[G.C.C.](https://twitter.com/elalaCorrea) is a fourth-year graduate fellow taking pictures of atoms. Likes electrons, microscopes, and pandesal.
"""
[[extra.authors]]
name = "Rolph Recto"
link = "https://twitter.com/rolphrecto"
[[extra.authors]]
name = "Gabriela Calinao Correa"
link = "https://twitter.com/elalaCorrea"
+++


Loop Invariant Code Motion hoists what doesn't need to be in the loop (invariant code) out of the loop. This optimization cuts down the number of instructions executed, by ensuring unnecessary repetition is avoided. Our implementation first identifies movable components, then iteratively moves them. 

Skip to the end to optimize your very own `bril` program!

# Loop

All loops considered here are **natural loops**. That is, a cycle with one entry and a **back-edge**. Back-edges are defined as an edge $A \longrightarrow B$ for tail $A$ and head $B$, such that $B$ dominates $A$.  Natural loops are then defined as the smallest set of vertices $L$ with $A,B \el L$ such that for each vertex $v \el L$ we have $v=B$ or PREDS($v$)$\subseteq L$.

### Detection

To find the loop invariant code, first we must detect all natural loops.

```python
def natloops(blocks): #input backedge
    pred,succ = edges(blocks)
    dom = get_dom(succ,list(blocks.keys())[0])
    for source,sink in get_backedges(succ,dom):
        yield loopsy(source,sink,pred) # natloops

def loopsy(source,sink,predecessors):
    worklist = [source]
    loop = set()
    while len(worklist)>0:
        current = worklist.pop()
        pr = predecessors[current]
        for p in pr:
            if not(p in loop or p==sink):
                loop.add(p)
                worklist.append(p)
    loop.add(sink)
    loop.add(source)
    return loop
```

### Invariance
Code may be marked as loop invariant if

# Motion


Now we know how to spot what can move, let's move it!

### Loop in a Loop in a Loop

#### How do we fix this?

### Super nested loops

# Try it!
Demo for people to see results on their own `bril` code
<!---eof--->
