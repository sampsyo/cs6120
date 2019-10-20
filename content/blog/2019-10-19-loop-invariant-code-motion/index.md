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

All loops considered here are **natural loops**—that is, a cycle with one entry and a **back-edge**. Back-edges are defined as an edge $A \longrightarrow B$ for tail $A$ and head $B$, such that $B$ dominates $A$.  Natural loops are then defined as the smallest set of vertices $L$ with $A,B \el L$ such that for each vertex $v \el L$ we have $v=B$ or PREDS($v$)$\subseteq L$.

In essence, a back-edge is what brings us from the end $A$ of the loop back to the beginning $B$. If the loop has more than one entry, it is not a natural loop. Below is an illustration of a natural loop and its back-edge. 

### Detecting natural loops

To find the loop invariant code, first we must detect all natural loops. To accomplish this, we make use of control flow graphs from `cfg.py` and dominator trees from `dom.py` within our three functions. Back-edges are identified with `get_backedges`, taking in a map of successors and the dominator tree. `loopsy` finds the natural loop associated with an input back-edge. 

```python
### detect back-edges
def get_backedges(successors,domtree):
  backedges = set()
  for source,sinks in successors.items():
    for sink in sinks:
      if sink in domtree[source]:
        backedges.add((source,sink))

  return backedges


### get natural loops
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

### Detecting loop invariants
An instruction within a natural loop is marked loop-invariant if its arguments are defined outside of the natural loop. Alternately, if the instruction’s arguments are defined once—and that definition is loop invariant—then the instruction may be marked as loop-invariant.

To begin with, we find all reaching definitions

```python
### apply reaching definitions analysis
def reachers(blocks):
  rins, routs = df_worklist(blocks, ANALYSIS["rdef"])
  return rins,routs


### get variable information for reaching definitions
def reaching_def_vars(blocks, reaching_defs):
  rdef_vars = {}

  for blockname, rdefs_block in reaching_defs.items():
    block = blocks[blockname]
    block_rdef_vars = []
    for rdef_blockid, rdef_instr in rdefs_block:
      block_rdef_vars.append( \
          (rdef_blockid, rdef_instr, blocks[rdef_blockid][rdef_instr]["dest"]))

    rdef_vars[blockname] = block_rdef_vars

  return rdef_vars
```

which then gives us the information to detect loop-invariant instructions using `invloop`.

# Motion


Now we know how to spot what can move, let's move it!

### Loop in a Loop in a Loop

#### How do we fix this?

### Super nested loops

# Try it!
Demo for people to see results on their own `bril` code
<!---eof--->
