+++
title = "Out of the Loop!"
[extra]
latex = true
[[extra.authors]]
name = "Rolph Recto"
link = "https://twitter.com/rolphrecto"
bio = """
[Rolph Recto](https://twitter.com/rolphrecto) is a third-year graduate student studying
the intersection of programming languages, security, and distributed systems. Likes climbing, scifi, and halo-halo.
"""
[[extra.authors]]
name = "Gabriela Calinao Correa"
link = "https://twitter.com/elalaCorrea"
bio = """
[G.C.C.](https://twitter.com/elalaCorrea) is a fourth-year graduate fellow taking pictures of atoms. Likes electrons, microscopes, and pandesal. 
"""
+++

Loop Invariant Code Motion hoists what doesn't need to be in the loop---invariant code---out of the loop. This optimization cuts down the number of instructions executed, by making sure unecessary repetition is avoided. To implement this we first identified natural loops and the invariant code within them. After moving the invariant code outside of their original natural loop, 


# Loop


What is a **natural loop**?


A **backedge**

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

### eof
