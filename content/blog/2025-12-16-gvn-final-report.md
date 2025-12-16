+++
title = "Optimizing Bril with Global Value Numbering: Final Report"
[extra]
latex = true
bio = """
Jake Hyun, Tobi Weinberg, and Adnan Armouti are CS PhD students at Cornell Tech.
"""
[[extra.authors]]
name = "Jake Hyun"
[[extra.authors]]
name = "Tobi Weinberg"
[[extra.authors]]
name = "Adnan Armouti"
+++

# Optimizing Bril with Global Value Numbering

*By Jake Hyun, Tobi Weinberg, and Adnan Armouti*

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Design and Implementation](#2-design-and-implementation)
   - [2.1 GVN Algorithm](#21-gvn-algorithm)
   - [2.2 Pruned SSA Implementation](#22-pruned-ssa-implementation)
   - [2.3 SSA Copy Propagation](#23-ssa-copy-propagation)
   - [2.4 Dominator-Tree Copy Elimination](#24-dominator-tree-copy-elimination)
   - [2.5 Fixed Parallel Copy Sequentialization](#25-fixed-parallel-copy-sequentialization)
   - [2.6 Real Phi Node Representation](#26-real-phi-node-representation)
   - [2.7 Copy Coalescing](#27-copy-coalescing)
   - [2.8 Phase 2 Optimizations: SSA Overhead Elimination](#28-phase-2-optimizations-ssa-overhead-elimination)
3. [Hardest Parts to Get Right](#3-hardest-parts-to-get-right)
4. [Empirical Evaluation](#4-empirical-evaluation)
5. [Generative AI Usage](#5-generative-ai-usage)
6. [Conclusion](#6-conclusion)
7. [Comparison with Cooper/Briggs Paper](#7-comparison-with-cooperbriggs-paper)

---

## 1. Project Goal

The goal of this project was to implement **Global Value Numbering (GVN)**, a powerful compiler optimization that eliminates redundant computations by identifying expressions that compute the same value. GVN is more powerful than local value numbering because it operates across basic block boundaries using the Static Single Assignment (SSA) form and dominator tree traversal.

Specific objectives included:

1. **Correctness**: The optimized programs must produce identical outputs to the original unoptimized programs for all inputs.
2. **Effectiveness**: The optimization should reduce both static (code size) and dynamic (runtime) instruction counts.
3. **Faithfulness to Literature**: Implement GVN using real phi nodes (as described in the original Cooper & Briggs algorithm) rather than simpler alternatives like Bril's get/set encoding.
4. **Minimal Overhead**: The SSA conversion and lowering process should not introduce more instructions than it removes.

---

## 2. Design and Implementation

The implementation consists of several interconnected components that work together to perform effective global value numbering. The overall pipeline is:

```
Input Program
    ↓
SSA Conversion (to_ssa.py with Pruned SSA)
    ↓
GVN Pass (gvn.py - dominator tree walk)
    ↓
SSA Copy Propagation (in SSA form)
    ↓
Dominator-Tree Copy Elimination (in SSA form)
    ↓
SSA Lowering (from_ssa.py with fixed copy sequentialization)
    ↓
Dead Code Elimination (tdce.py)
    ↓
Optimized Program
```

### 2.1 GVN Algorithm {#21-gvn-algorithm}

#### 2.1.1 Core Data Structures

The GVN implementation (`gvn.py`) uses several key data structures:

**Value Numbers (VN)**: Immutable identifiers assigned to values. Two expressions receive the same VN if and only if they compute the same value.

```python
@dataclass(frozen=True)
class VN:
    n: int
```

**GVNContext**: Maintains the state during GVN traversal:

```python
class GVNContext:
    def __init__(self):
        self.vn_counter = 0                    # Next VN to allocate
        self.expr2vn_stack: List[Dict] = []    # Scoped expression → VN mapping
        self.vn_of: Dict[str, VN] = {}         # SSA name → VN (global)
```

The expression table uses a **stack of scopes** to handle the dominator tree traversal. When we descend into a dominated block, we push a new scope. Expressions found in dominated blocks are visible to further descendants but not to siblings or ancestors.

#### 2.1.2 Expression Keys

Expressions are canonicalized into hashable keys for the expression table:

```python
def expr_key(instr: Dict, arg_vns: List[int]) -> Optional[Tuple]:
    op = instr.get("op")
    ty = type_key(instr.get("type"))

    if op == "const":
        return ("const", ty, instr.get("value"))

    # Normalize gt/ge → lt/le with swapped args
    if op in REL_FLIPS:
        op = REL_FLIPS[op]
        arg_vns = list(reversed(arg_vns))

    if op in PURE_BIN or op in PURE_UN:
        key_args = arg_vns[:]
        # Canonicalize commutative ops
        if op in COMMUTATIVE and len(key_args) == 2:
            if key_args[0] > key_args[1]:
                key_args = [key_args[1], key_args[0]]
        return (op, ty, *key_args)

    return None
```

Key design decisions:
- **Commutative normalization**: For operations like `add`, `mul`, `and`, we sort arguments by VN so that `a + b` and `b + a` produce the same key.
- **Relational normalization**: We flip `gt`/`ge` to `lt`/`le` by swapping arguments, reducing the number of distinct expression patterns.
- **Type inclusion**: Types are included in the key because operations on different types (e.g., `int` vs `float`) are not interchangeable.

#### 2.1.3 Dominator Tree Walk

The core GVN algorithm performs a depth-first traversal of the dominator tree:

```python
def gvn_walk(block_name: str, ctx: GVNContext, env: Dict[str, VN],
             val2rep: Dict[Tuple, str], bmap: Dict[str, dict],
             dom_tree: Dict[str, List[str]], ...):

    ctx.push_scope()

    # Process phi nodes first
    for phi in phi_instructions:
        process_phi(phi, ctx, env, val2rep, has_backedge)

    # Process regular instructions
    for instr in regular_instructions:
        if is_pure(instr):
            # Look up expression, replace with id if redundant
            key = expr_key(instr, arg_vns)
            existing = ctx.lookup_expr(key)
            if existing:
                replace_with_id(instr, existing)
            else:
                ctx.insert_expr(key, new_vn)
        else:
            # Effectful: fresh VN, clear expression cache
            ctx.clear_all_scopes()

    # Recurse into dominated children with copied environment
    for child in dom_tree[block_name]:
        gvn_walk(child, ctx, env.copy(), val2rep.copy(), ...)

    ctx.pop_scope()
```

The environment (`env`) and value-to-representative mapping (`val2rep`) are **copied** when descending into children. This ensures that bindings in one branch don't affect sibling branches.

#### 2.1.4 Conservative Handling

The implementation is conservative in several ways:

1. **Effectful Operations**: Any instruction with side effects (memory operations, function calls, I/O) clears all expression scopes:
   ```python
   if not is_pure(instr):
       ctx.clear_all_scopes()
   ```

2. **Loop Backedges**: Phi nodes at loop headers receive fresh value numbers because their values may change on each iteration:
   ```python
   if has_backedge:
       vn = ctx.new_vn()  # Conservative: fresh VN for loop header phis
   ```

3. **Division Excluded**: Integer division is excluded from pure operations because speculating a division could trap on divide-by-zero.

### 2.2 Pruned SSA Implementation {#22-pruned-ssa-implementation}

#### 2.2.1 The Problem with Minimal SSA

The initial implementation used **minimal SSA**, which places phi nodes at all dominance frontiers of variable definitions. This caused severe code bloat:

| Benchmark | Original | After Minimal SSA | Bloat |
|-----------|----------|-------------------|-------|
| fizz-buzz.bril | 57 | 149 | +162% |
| Loop-heavy programs | N | ~2.5N | +150% |

The root cause: minimal SSA creates phi nodes for **every** variable at dominance frontiers, even if the variable is never used after that point.

#### 2.2.2 Pruned SSA Solution

**Pruned SSA** only places phi nodes for variables that are actually **live** at the join point. This requires liveness analysis:

```python
def compute_liveness(blocks, next_map):
    """Compute live-in and live-out sets for each block using dataflow analysis.

    A variable is live at a point if it may be used before being redefined.
    """
    # Compute use/def sets per block
    use_sets = {}
    def_sets = {}
    for b in blocks:
        uses, defs = set(), set()
        for ins in b["instrs"]:
            # Uses before def count as upward-exposed uses
            for arg in ins.get("args", []):
                if isinstance(arg, str) and arg not in defs:
                    uses.add(arg)
            if "dest" in ins:
                defs.add(ins["dest"])
        use_sets[b["name"]] = uses
        def_sets[b["name"]] = defs

    # Iterate until fixpoint (backward dataflow)
    live_in = {b["name"]: set() for b in blocks}
    live_out = {b["name"]: set() for b in blocks}

    changed = True
    while changed:
        changed = False
        for b in reversed(blocks):
            name = b["name"]
            # live_out = union of live_in of successors
            new_out = set()
            for succ in next_map.get(name, []):
                new_out |= live_in[succ]

            # live_in = use ∪ (live_out - def)
            new_in = use_sets[name] | (new_out - def_sets[name])

            if new_in != live_in[name] or new_out != live_out[name]:
                changed = True
                live_in[name] = new_in
                live_out[name] = new_out

    return live_in, live_out
```

The phi placement algorithm is then modified:

```python
def place_phis(blocks, next_map, df_map, defs, vtype, live_in=None):
    """Place phi nodes using pruned SSA algorithm."""
    for v, def_blocks in defs.items():
        W = list(def_blocks)
        while W:
            n = W.pop()
            for y in df_map.get(n, set()):
                key = (y, v)
                if key not in have_phi:
                    # PRUNED SSA: Only place phi if variable is live at y
                    if live_in is not None and v not in live_in.get(y, set()):
                        continue  # Skip - variable not live here

                    # Place phi node...
```

#### 2.2.3 Impact of Pruned SSA

| Benchmark | Phi Nodes (Minimal) | Phi Nodes (Pruned) | Reduction |
|-----------|---------------------|--------------------|-----------|
| fizz-buzz.bril | 73 | 1 | 98.6% |
| loopfact.bril | 12 | 2 | 83.3% |
| Average | N | ~0.15N | 85%+ |

This dramatically reduced the SSA overhead, making the optimization effective.

### 2.3 SSA Copy Propagation {#23-ssa-copy-propagation}

#### 2.3.1 Design

In SSA form, copy propagation is **always safe** because each variable has exactly one definition. This means we can aggressively propagate copies without worrying about reaching definitions from different paths.

```python
def ssa_copy_propagation(func: dict):
    """Propagate copies while in SSA form.

    In SSA form, each variable has exactly ONE definition, so copy propagation
    is safe. We can propagate copies even into phi arguments because in SSA,
    every definition dominates all its uses.
    """
    # Pass 1: Build copy map (dest -> src) for all id instructions
    # Exclude phi destinations since they have special merge semantics
    phi_dests: Set[str] = set()
    for instr in instrs:
        if instr.get("op") == "phi":
            phi_dests.add(instr.get("dest"))

    copy_map: Dict[str, str] = {}
    for instr in instrs:
        if instr.get("op") == "id":
            dest = instr.get("dest")
            src = instr["args"][0]
            if dest not in phi_dests:
                copy_map[dest] = src

    # Helper: follow copy chain to ultimate source
    def ultimate_src(v: str) -> str:
        visited = set()
        while v in copy_map and v not in visited:
            visited.add(v)
            v = copy_map[v]
        return v

    # Pass 2: Rewrite uses (INCLUDING phi args)
    for instr in instrs:
        if "args" in instr:
            instr["args"] = [ultimate_src(a) for a in instr["args"]]
```

#### 2.3.2 Key Insight: Phi Argument Propagation

A subtle but important point: we **can** propagate copies into phi arguments in SSA form. This is safe because in SSA, every definition dominates all its uses. If `x.1 = id y.1`, then wherever `x.1` appears (including in phi nodes in successor blocks), we can replace it with `y.1`.

This is a key advantage of performing copy propagation **before** SSA lowering.

### 2.4 Dominator-Tree Copy Elimination {#24-dominator-tree-copy-elimination}

#### 2.4.1 Design

After SSA copy propagation, there may still be `id` instructions that weren't eliminated because they define phi destinations or have other uses. The dominator-tree copy elimination pass walks the dominator tree with a **rename environment**:

```python
def dom_tree_copy_elimination(func: dict):
    """Eliminate copies using dominator-tree-based propagation.

    Walks the dominator tree carrying a rename environment. For each `id`
    instruction, we establish a rename from dest to source and drop the
    instruction.
    """
    def walk(block_name: str, rename: Dict[str, str]):
        block = bmap[block_name]
        new_instrs = []

        for instr in block.get("instrs", []):
            op = instr.get("op")

            # Rewrite arguments using rename map (but NOT phi args)
            if "args" in instr and op != "phi":
                instr["args"] = [canon(a, rename) for a in instr["args"]]

            # Handle id instructions
            if op == "id":
                dest = instr.get("dest")
                if dest not in phi_dests:
                    src = canon(args[0], rename)
                    rename[dest] = src
                    continue  # Drop the id instruction

            new_instrs.append(instr)

        block["instrs"] = new_instrs

        # Recurse with copied environment
        for child in dom_tree.get(block_name, []):
            walk(child, rename.copy())
```

#### 2.4.2 Why This is Safe

The key insight is that renames established in a dominator are valid in all dominated blocks. By copying the environment before recursing into children, we ensure that renames from one branch don't leak into sibling branches.

**Example:**
```
     B0: x = 5
    /  \
   B1   B2
   y=x  z=x
```

If we're at B0 and establish `x → 5`, both B1 and B2 see this rename. But renames in B1 don't affect B2 because they receive separate copies of the environment.

#### 2.4.3 Preserving Phi Destinations

We explicitly preserve phi destinations because they are **merge points**. Even if a phi produces `x = phi(y, z)` where `y` and `z` have the same value, we can't eliminate the phi — it defines a new SSA name that may be used later.

### 2.5 Fixed Parallel Copy Sequentialization {#25-fixed-parallel-copy-sequentialization}

#### 2.5.1 The Problem

When lowering phi nodes to copies, we face the **parallel copy problem**. Phi nodes represent parallel assignments:

```
# At block B with predecessors P1, P2:
x = phi(a from P1, b from P2)
y = phi(c from P1, d from P2)
```

These become parallel copies at the end of each predecessor:
```
# End of P1:
(x, y) := (a, c)  # Parallel!
```

If we naively sequentialize:
```
x = a
y = c
```

This is fine. But consider:
```
(x, y) := (y, x)  # Swap!
```

Naive sequentialization fails:
```
x = y   # x gets y's value
y = x   # y gets... x's new value (wrong!)
```

#### 2.5.2 The Algorithm

The fixed algorithm handles dependencies correctly:

```python
def _parallel_copies_to_seq(copies):
    """Convert parallel copies to sequential copies, handling dependencies.

    A copy (d, s) can be emitted safely if:
    1. Source s won't be overwritten (s is not a dest in remaining copies), AND
    2. Dest d is not used as a source by another remaining copy
    """
    seq = []
    temp_i = 0

    while copies:
        dests = {d for (d, s, t) in copies}
        sources = {s for (d, s, t) in copies}

        # Find a safe copy
        picked = None
        for i, (d, s, t) in enumerate(copies):
            source_safe = (s not in dests) or (s == d)
            dest_safe = (d not in sources) or (d == s)
            if source_safe and dest_safe:
                picked = i
                break

        if picked is not None:
            d, s, t = copies.pop(picked)
            if d != s:
                seq.append({"op": "id", "dest": d, "type": t, "args": [s]})
            continue

        # No safe copy - break cycle with temp
        # Find a value that's both a dest AND a source
        cycle_var = None
        for (d, s, t) in copies:
            if d in sources and d in dests:
                cycle_var = d
                cycle_type = t
                break

        # Save to temp and update copies
        tmp = f"__phi_tmp{temp_i}"
        temp_i += 1
        seq.append({"op": "id", "dest": tmp, "type": cycle_type, "args": [cycle_var]})

        copies = [(d, tmp if s == cycle_var else s, t) for (d, s, t) in copies]

    return seq
```

#### 2.5.3 The Bug We Fixed

The original algorithm would loop infinitely on certain copy patterns because it didn't correctly identify when to break a cycle. Specifically, it failed when:

1. A destination was used as a source by another copy
2. Multiple variables formed a cycle

The fix correctly identifies cycle members (variables that are both sources and destinations) and breaks the cycle by saving one to a temporary.

**Example of the fix in action:**
```
# Parallel: (x, y, z) := (y, z, x)  # 3-way cycle
# Step 1: No safe copy (all dests are sources)
# Step 2: Save x to tmp: tmp = x
# Step 3: Update: (x, y, z) := (y, z, tmp)
# Step 4: Now z := tmp is safe, emit it
# Step 5: Now y := z is safe, emit it
# Step 6: Now x := y is safe, emit it
# Result: tmp=x; z=tmp; y=z; x=y
```

### 2.6 Real Phi Node Representation {#26-real-phi-node-representation}

#### 2.6.1 Why Real Phi Nodes?

The original GVN paper by Cooper and Briggs uses real phi nodes in SSA form. Bril provides an alternative approach using `get`/`set` operations with labels, which is easier to implement but doesn't faithfully represent SSA.

We chose to implement **real phi nodes** for several reasons:

1. **Algorithmic Fidelity**: The GVN algorithm naturally operates on phi nodes as first-class instructions
2. **Value Numbering of Phis**: We can assign value numbers to phi nodes and detect redundant phis
3. **Learning Value**: Understanding real SSA conversion is pedagogically important

#### 2.6.2 Phi Node Structure

Our phi nodes have the following structure:

```json
{
  "op": "phi",
  "dest": "x.3",
  "type": "int",
  "args": ["x.1", "x.2"],
  "labels": ["B1", "B2"]
}
```

The `args` and `labels` arrays are parallel — `args[i]` is the value when control comes from `labels[i]`.

#### 2.6.3 Phi Node Processing in GVN

Phi nodes are processed specially in the GVN walk:

```python
def process_phi(phi: dict, ctx: GVNContext, env: Dict[str, VN],
                val2rep: Dict[Tuple, str], has_backedge: bool) -> VN:
    dest = phi["dest"]
    args = phi.get("args", [])

    if has_backedge:
        # Conservative: fresh VN for loop header phis
        return ctx.new_vn()

    # Map phi args to their VNs
    arg_vns = [ctx.get_vn(arg).n for arg in args]

    # Trivial phi: all args have same VN
    if all(v == arg_vns[0] for v in arg_vns):
        return VN(arg_vns[0])

    # Check for redundant phi (same arg VN pattern)
    phi_key = ("phi", type_key(ty), tuple(arg_vns))
    existing = ctx.lookup_expr(phi_key)
    if existing:
        return existing

    vn = ctx.new_vn()
    ctx.insert_expr(phi_key, vn)
    return vn
```

This allows GVN to:
- Recognize trivial phis (all args same) and propagate the value
- Detect redundant phis (same pattern of arg VNs)
- Be conservative at loop headers

### 2.7 Copy Coalescing {#27-copy-coalescing}

#### 2.7.1 The Problem: Phi Lowering Overhead

After implementing all previous optimizations, we discovered that **63 out of 114 benchmarks were still worse** in both static and dynamic instruction count. The root cause was **phi lowering overhead**.

When converting out of SSA form, each phi node becomes multiple `id` (copy) instructions:

**Example: triangle.bril**
```
Original:    14 instructions, 0 id copies
After SSA:   12 instructions (2 phis)
After GVN:   12 instructions (phis intact)
After lower: 17 instructions, 4 id copies   ← +5 instructions!
```

The copies introduced are:
1. **Entry copies**: Initialize phi variables with their first-iteration values
2. **Backedge copies**: Update phi variables for the next iteration

```bril
@tri(n: int): int {
  one.1: int = const 1;
  t.1: int = const 0;
  c.2: int = id one.1;      ← Entry copy for c phi
  t.2: int = id t.1;        ← Entry copy for t phi
.c:
  cond_c.1: bool = le c.2 n;
  br cond_c.1 .c_le .c_gt;
.c_gt:
  ret t.2;
.c_le:
  t.3: int = add t.2 c.2;
  c.3: int = add c.2 one.1;
  c.2: int = id c.3;        ← Backedge copy for c phi
  t.2: int = id t.3;        ← Backedge copy for t phi
  jmp .c;
}
```

#### 2.7.2 Why Previous Copy Elimination Failed

Our SSA copy propagation and dominator-tree copy elimination couldn't remove these copies because:

1. The copies are **required for correctness** in naive phi lowering
2. The variables use the **same name across iterations** (`c.2`, `t.2`)
3. They appear to be live across the entire loop

#### 2.7.3 The Solution: Copy Coalescing

**Copy coalescing** recognizes that two SSA names can share the same variable if their live ranges don't interfere (overlap).

**Key Insight:**
```bril
c.3: int = add c.2 one.1;  // c.3 STARTS here
c.2: int = id c.3;         // This copy is unnecessary!
```

Since `c.2`'s last use is in the `add` instruction, and `c.3` starts immediately after, their live ranges **don't interfere**. We can rename `c.3 → c.2` and eliminate the copy!

#### 2.7.4 Trivial Copy Coalescing

We first implemented a simple heuristic that catches many cases without full liveness analysis:

For a copy `x = id y`:
- If `y` is defined immediately before the copy (in the same block)
- And `y` has no other uses after its definition
- Then rename `y → x` in the defining instruction and remove the copy

```python
def trivial_copy_coalescing(func: dict):
    """Eliminate copies where source is defined immediately before and unused after."""
    instrs = func.get("instrs", [])

    # Find copy candidates: y defined right before x = id y
    for i, instr in enumerate(instrs):
        if instr.get("op") != "id":
            continue
        dest, src = instr.get("dest"), instr["args"][0]

        if i > 0 and instrs[i-1].get("dest") == src:
            # Check if src is used anywhere else
            src_used_elsewhere = any(
                src in other.get("args", [])
                for j, other in enumerate(instrs) if j != i
            )
            if not src_used_elsewhere:
                # Rename: change y to x in the defining instruction
                instrs[i - 1]["dest"] = dest
                instrs[i] = None  # Mark for removal

    func["instrs"] = [i for i in instrs if i is not None]
```

#### 2.7.5 Full Copy Coalescing with Liveness

For copies not caught by trivial coalescing, we implemented full liveness-based coalescing:

**Step 1: Compute Instruction-Level Liveness**

```python
def compute_instruction_liveness(func: dict) -> Tuple[Dict, Dict]:
    """Compute live-in and live-out sets for each instruction."""
    # Build instruction-level CFG
    # Run backward dataflow to fixpoint
    # Return live_in[instr_idx] and live_out[instr_idx]
```

**Step 2: Build Interference Information**

Two variables **interfere** if they are both live at the same program point:

```python
def full_copy_coalescing(func: dict):
    live_in, live_out = compute_instruction_liveness(func)

    for idx, instr in enumerate(instrs):
        if instr.get("op") != "id":
            continue
        dest, src = instr.get("dest"), instr["args"][0]

        # Check interference: is dest live at src's definition?
        src_def_idx = find_def_index(src)
        if src_def_idx is not None:
            live_at_src_def = live_out.get(src_def_idx, set())
            if dest not in live_at_src_def:
                # No interference - can coalesce!
                rename_map[dest] = src
```

**Step 3: Apply Renames**

```python
# Rename all uses of dest to src
for instr in instrs:
    if "args" in instr:
        instr["args"] = [rename_map.get(a, a) for a in instr["args"]]
    if instr.get("dest") in rename_map:
        instr["dest"] = rename_map[instr["dest"]]
```

#### 2.7.6 Critical Bug Fix: Function Parameters

Our initial implementation had a subtle bug: it would rename function parameters on subsequent coalescing iterations, causing undefined variable errors.

**Example of the bug:**
```
Iteration 1: input.1 = id input  →  rename input.1 → input ✓
Iteration 2: input.2 = id input  →  rename input → input.2 ✗
```

After iteration 2, uses of `input` (the parameter) would point to `input.2`, which is defined later!

**The Fix:** Never coalesce when the source is a function parameter:

```python
param_names = {arg["name"] for arg in func.get("args", [])}

for dest, src in copies:
    if src in param_names:
        continue  # Never rename away from a parameter
```

#### 2.7.7 Impact of Copy Coalescing

| Metric | Before Coalescing | After Coalescing | Change |
|--------|-------------------|------------------|--------|
| Static Ratio (GM) | 0.9293x | **0.8477x** | +8.8% better |
| Dynamic Ratio (GM) | 0.9152x | **0.8167x** | +9.6% better |
| Benchmarks worse (both) | 63/114 | **30/114** | -52% |
| Benchmarks improved (static) | 38/114 | **50/114** | +32% |

Copy coalescing transformed the optimizer from a net negative to a clear positive on the majority of benchmarks.

### 2.8 Phase 2 Optimizations: SSA Overhead Elimination {#28-phase-2-optimizations-ssa-overhead-elimination}

After implementing copy coalescing (Section 2.7), **30 out of 114 benchmarks** were still worse in both static and dynamic instruction count. Further analysis revealed that the remaining overhead came from specific patterns in phi lowering. We implemented three additional optimizations to address these patterns.

#### 2.8.1 Root Cause Analysis

The remaining overhead came from three sources:

1. **Parameter Copies**: When a function parameter feeds into a phi node, phi lowering creates a copy like `x.1 = id x`
2. **Entry Block Constant Copies**: When a constant initializes a phi variable in the entry block
3. **Backedge Copies**: Copies at loop backedges that couldn't be coalesced

#### 2.8.2 Parameter Renaming

We observed a recurring pattern where function parameters were immediately copied into phi-destination SSA names:

```bril
@main(x: int) {
  x.1: int = id x;      ← Unnecessary copy from parameter
  ...
}
```

This copy is required by naive phi lowering, but it's wasteful. The solution is to rename the function parameter itself to match the phi destination, eliminating the copy altogether:

```python
def parameter_renaming(func: dict):
    """Rename function parameters to match their phi-destination SSA names.

    IMPORTANT: Only rename if:
    1. The parameter is ONLY used by the copy (use count == 1)
    2. The copy is in the ENTRY BLOCK (before any labels) - not inside a loop!
    """
    instrs = func.get("instrs", [])
    args = func.get("args", [])
    if not args:
        return

    param_names = {arg["name"] for arg in args}

    # Count uses of each parameter
    param_use_count = {name: 0 for name in param_names}
    for instr in instrs:
        for arg in instr.get("args", []):
            if arg in param_names:
                param_use_count[arg] += 1

    # Find the first label to determine entry block boundary
    first_label_idx = len(instrs)
    for i, instr in enumerate(instrs):
        if "label" in instr:
            first_label_idx = i
            break

    # Only consider copies BEFORE the first label (entry block)
    for i, instr in enumerate(instrs):
        if i >= first_label_idx:
            break  # Stop if past entry block

        if instr.get("op") == "id" and "dest" in instr:
            src = instr.get("args", [])[0]
            if src in param_names and param_use_count[src] == 1:
                # Rename parameter to match destination
                rename_parameter(func, src, instr["dest"])
                remove_copy(i)
```

**Critical Safety Check:** The entry block boundary check prevents renaming parameters that are used inside loops. Without this check, the optimizer would incorrectly rename loop-carried copies, breaking programs like `1dconv.bril`.

#### 2.8.3 Constant Copy Fusion

Similarly, constants that were immediately copied also represented unnecessary overhead:

```bril
i.1: int = const 0;
res.2: int = id i.1;    ← Unnecessary copy from constant
```

When a constant is only used once (by a copy instruction), we can fuse them by directly changing the constant's destination to the final SSA name:

```python
def const_copy_fusion(func: dict):
    """Fuse const definitions with their single-use copies."""
    # Find constants used only once
    use_counts = count_all_uses(func)

    for i, instr in enumerate(instrs):
        if instr.get("op") == "id":
            src = instr["args"][0]
            if is_const(src) and use_counts[src] == 1:
                # Change const destination to copy destination
                const_instr["dest"] = instr["dest"]
                remove_copy(i)
```

#### 2.8.4 Phi Destination Propagation

A third pattern emerges when we trace the flow from entry block definitions through phi nodes:

```bril
res.1: int = const 0;
res.2: int = id res.1;    ← Entry copy for phi
.loop:
  res.2: int = phi res.2 from entry, res.3 from body
```

The entry definition (`res.1`) exists only to serve the phi node. If that definition is only used by the entry copy, we can rename the definition directly to the phi destination name and eliminate the copy:

```python
def phi_destination_propagation(func: dict):
    """Propagate phi destination names to entry block definitions."""
    # For each phi, if the entry argument is defined in the entry block
    # and only used by the phi, rename the definition to use phi's dest
    for phi in phi_instructions:
        entry_arg = phi["args"][entry_index]
        if defined_in_entry_block(entry_arg) and use_count(entry_arg) == 1:
            rename_def(entry_arg, phi["dest"])
```

#### 2.8.5 TDCE Fix for Function Parameters

**The Bug:** After parameter renaming, TDCE would incorrectly remove code that depended on renamed parameters.

**Example:**
```bril
@convolve(arr.3: ptr<int>, ...) {  ← Parameter renamed from arr
  loc.1: ptr<int> = ptradd arr.3 offset;
  ...
}
```

TDCE didn't know `arr.3` was a parameter and removed the `ptradd` instruction.

**The Fix:** Add function parameter names to the `global_uses` set in TDCE:

```python
def trivial_dce_func(func):
    # ... existing code ...
    global_uses = find_globals_used_elsewhere(blocks)

    # Include function parameters as globally needed variables
    # This prevents removing code that uses renamed parameters
    for arg in func.get("args", []):
        global_uses.add(arg["name"])
```

#### 2.8.6 Impact of Phase 2 Optimizations

| Metric | After Phase 1 | After Phase 2 | Change |
|--------|---------------|---------------|--------|
| Static Ratio (GM) | 0.8474x | **0.8397x** | +0.9% better |
| Dynamic Ratio (GM) | 0.8165x | **0.8142x** | +0.3% better |
| Benchmarks worse (both) | 30/114 | **22/114** | -27% |
| Benchmarks improved (static) | 47/114 | **53/114** | +13% |

These optimizations further reduced the number of regressions while maintaining 100% correctness on all 114 benchmarks.

---

## 3. Hardest Parts to Get Right

### 3.1 SSA Overhead: The 162% Bloat Problem

**The Challenge**: Our initial implementation caused the optimizer to **increase** code size by up to 162% on some benchmarks. This was unacceptable — an optimizer should never make code worse.

**Root Cause Analysis**: We spent significant time tracing through the pipeline to understand why. The culprit was minimal SSA placing 73 phi nodes in `fizz-buzz.bril` (a 57-instruction program).

**The Solution**: Implementing pruned SSA with liveness analysis. This required:
1. Implementing a full backward dataflow analysis for liveness
2. Modifying phi placement to check liveness
3. Carefully handling the interaction between liveness and SSA renaming

**Key Insight**: The difference between minimal and pruned SSA is often glossed over in textbooks, but it's critical for practical compilers. Dead variables at join points don't need phi nodes.

### 3.2 Parallel Copy Sequentialization Infinite Loop

**The Challenge**: The SSA lowering pass would hang indefinitely on certain programs.

**Root Cause**: The algorithm for converting parallel copies to sequential copies had a bug in cycle detection. When a destination was used as a source by another copy, the algorithm couldn't make progress.

**The Solution**: Complete rewrite of `_parallel_copies_to_seq()` with proper cycle detection:
1. Identify safe copies (source won't be overwritten, dest not needed)
2. If no safe copy, find a cycle variable (both source and dest)
3. Break the cycle by saving to a temporary
4. Update remaining copies to use the temporary

**Key Insight**: Parallel copy sequentialization is a classic problem, but subtle to get right. The solution is essentially graph-based: find a topological order if acyclic, break cycles with temps otherwise.

### 3.3 Copy Propagation Soundness

**The Challenge**: Aggressive copy propagation can be unsound if done after SSA lowering, because copies inserted by phi lowering are **necessary** for correctness.

**Example of the bug**:
```
# After phi lowering:
# End of B1:
x = a    # Copy for phi
jmp B3

# B3:
use(x)
```

If we propagate `x → a` and eliminate the copy, the program breaks because `a` might not be available in B3 (it came from a different predecessor).

**The Solution**: Perform copy propagation **while still in SSA form**, before phi lowering. In SSA, every definition dominates all its uses, so copy propagation is always safe.

### 3.4 Entry Block Creation for Label-First Functions

**The Challenge**: Some Bril functions start with a label (indicating the first block is a loop header or join point). This caused incorrect dominator analysis.

**The Solution**: Always create an entry block `B0` if the function starts with a label:

```python
def form_blocks(instrs):
    blocks = []
    state = {'label': None, 'instrs': [], 'next_id': 0}

    # Always create an entry block if function starts with a label
    if instrs and 'label' in instrs[0]:
        blocks.append({"name": "B0", "label": None, "instrs": []})
        state['next_id'] = 1
    # ...
```

This ensures proper CFG structure for dominator analysis.

### 3.5 Copy Coalescing and the 63-Benchmark Regression

**The Challenge**: After implementing all SSA optimizations, we discovered that **63 out of 114 benchmarks** were still worse in both static and dynamic instruction count. The optimizer was making most programs worse!

**Root Cause**: Phi lowering introduces `id` (copy) instructions that weren't being eliminated:
- Entry copies initialize phi variables before loops
- Backedge copies update phi variables for the next iteration

These copies appeared "live" across iterations because they used the same variable names.

**The Solution**: Implement copy coalescing with liveness analysis:
1. Compute instruction-level live ranges
2. For each copy `x = id y`, check if `x` interferes with `y`
3. If no interference, rename `x → y` everywhere and remove the copy

**Critical Bug**: The first implementation accidentally renamed function parameters, breaking later iterations. The fix: never coalesce away from a function parameter.

**Result**: This reduced "worse in both" from 63/114 to 30/114 benchmarks, and improved the static ratio from 0.93x to **0.85x**.

---

## 4. Empirical Evaluation

### 4.1 Methodology

**Benchmarks**: We evaluated on 123 Bril benchmarks across 5 categories:
- `core`: 68 general-purpose benchmarks
- `float`: 18 floating-point arithmetic benchmarks
- `mem`: 31 memory operation benchmarks
- `mixed`: 4 combined operation benchmarks
- `long`: 2 large/complex programs

**Correctness**: Every optimized program was verified to produce identical output to the original for all provided inputs.

**Metrics**:
- **Static Instruction Count**: Total instructions in the program (code size)
- **Dynamic Instruction Count**: Instructions executed at runtime (performance)
- **Geometric Mean Ratio**: GM of (optimized/original) across benchmarks

### 4.2 Results Summary

| Metric | Value |
|--------|-------|
| Total Benchmarks | 123 |
| All Tests Pass | 100% |
| Static Instruction Ratio (GM) | **0.8411** (15.9% reduction) |
| Dynamic Instruction Ratio (GM) | **0.8170** (18.3% reduction) |
| Benchmarks with Static Improvement | 62/123 (50.4%) |
| Benchmarks with Dynamic Improvement | 62/123 (50.4%) |
| Benchmarks worse in both metrics | 22/123 (17.9%) |

### 4.3 Best Improvements

**Top 5 Static Improvements:**

| Benchmark | Original | Optimized | Ratio |
|-----------|----------|-----------|-------|
| ray-sphere-intersection.bril | 106 | 30 | 0.283x |
| pascals-row.bril | 41 | 18 | 0.439x |
| euclid.bril | 51 | 23 | 0.451x |
| sqrt_bin_search.bril | 44 | 20 | 0.455x |
| sum-sq-diff.bril | 66 | 30 | 0.455x |

**Top 5 Dynamic Improvements:**

| Benchmark | Original | Optimized | Ratio |
|-----------|----------|-----------|-------|
| ray-sphere-intersection.bril | 142 | 42 | 0.296x |
| sum-sq-diff.bril | 3038 | 1117 | 0.368x |
| sqrt_bin_search.bril | 744 | 288 | 0.387x |
| birthday.bril | 484 | 188 | 0.388x |
| pascals-row.bril | 146 | 57 | 0.390x |

### 4.4 Analysis by Category

| Category | Benchmarks | Static GM | Dynamic GM |
|----------|------------|-----------|------------|
| core | 68 | 0.8154 | 0.7879 |
| float | 18 | 0.7371 | 0.6948 |
| mem | 31 | 0.9506 | 0.9591 |
| mixed | 4 | 0.9728 | 0.9527 |
| long | 2 | 0.8870 | 0.7373 |

The `float` benchmarks show the best improvement (26.3% static, 30.5% dynamic) due to floating-point computations having many redundant expressions. The `core` benchmarks also show strong improvements (18.5% static, 21.2% dynamic). Memory-heavy benchmarks (`mem`) show minimal improvement because memory operations are effectful and cannot be optimized by GVN.

### 4.5 Comparison with Get/Set SSA Implementation

We compared our real phi node implementation against an alternative approach using Bril's `get`/`set` label-based syntax. While Bril supports `get`/`set` for SSA representation, the Cooper/Briggs paper assumes real phi nodes. We initially implemented GVN using `get`/`set`, then re-implemented it with real phi nodes for comparison.

| Metric | Our Implementation (Real Phis) | Get/Set Implementation | Improvement |
|--------|--------------------------------|------------------------|-------------|
| Total Static (62 common tests) | 2,262 | 2,313 | +51 instructions better |
| Static Reduction | 12.0% | 10.0% | +2.0% better |

The final phi node implementation outperforms the `get`/`set` approach by 51 instructions on the common benchmark set, primarily because real phis expose phi node structure to the GVN algorithm, enabling better redundancy detection and more precise dominator-tree analysis.

### 4.6 Visualization

The following figures are generated by `generate_figures.py`:

**GVN Optimization Results:**
1. **static_improvement_bar.png**: Bar chart showing top 20 benchmarks by static instruction reduction
2. **static_ratio_histogram.png**: Distribution of static instruction ratios across all benchmarks
3. **dynamic_improvement_bar.png**: Bar chart showing top 15 benchmarks by dynamic instruction reduction
4. **category_comparison.png**: Grouped bar chart comparing improvements by benchmark category
5. **static_vs_dynamic_scatter.png**: Scatter plot showing correlation between static and dynamic improvements
6. **summary_table.png**: Summary statistics table
7. **waterfall_savings.png**: Waterfall chart showing cumulative instruction savings

![Static improvement (top 20)](2025-12-16-gvn-final-report/static_improvement_bar.png)

![Static ratio distribution](2025-12-16-gvn-final-report/static_ratio_histogram.png)

![Dynamic improvement (top 15)](2025-12-16-gvn-final-report/dynamic_improvement_bar.png)

![Category comparison](2025-12-16-gvn-final-report/category_comparison.png)

![Static vs Dynamic scatter](2025-12-16-gvn-final-report/static_vs_dynamic_scatter.png)

![Results summary table](2025-12-16-gvn-final-report/summary_table.png)

![Waterfall savings](2025-12-16-gvn-final-report/waterfall_savings.png)

**SSA Overhead Analysis:**
8. **ssa_overhead_bar.png**: Bar chart comparing SSA overhead vs GVN savings per benchmark
9. **ssa_overhead_histogram.png**: Distribution of SSA overhead ratios
10. **ssa_vs_gvn_scatter.png**: Scatter plot of SSA overhead vs final GVN result
11. **ssa_pipeline_comparison.png**: Instruction counts at each pipeline stage
12. **ssa_summary_table.png**: SSA overhead and GVN recovery summary statistics

![SSA overhead vs savings](2025-12-16-gvn-final-report/ssa_overhead_bar.png)

![SSA overhead distribution](2025-12-16-gvn-final-report/ssa_overhead_histogram.png)

![SSA vs GVN scatter](2025-12-16-gvn-final-report/ssa_vs_gvn_scatter.png)

![SSA pipeline comparison](2025-12-16-gvn-final-report/ssa_pipeline_comparison.png)

![SSA summary table](2025-12-16-gvn-final-report/ssa_summary_table.png)

### 4.7 SSA Overhead Analysis

A key challenge in SSA-based optimization is the **overhead introduced by SSA conversion itself**. Converting to SSA form and back introduces phi nodes, which lower to copy instructions. This analysis quantifies that overhead and shows how our optimizations recover from it.

#### 4.7.1 SSA Conversion Overhead

When we convert a program to SSA form and then lower it back (without any GVN optimization), we observe:

| Metric | Value |
|--------|-------|
| Total Benchmarks Analyzed | 121 |
| Total Instructions (Original) | 8,089 |
| Total Instructions (After SSA) | 8,980 |
| **SSA Overhead** | **+891 instructions (+11.0%)** |
| SSA Overhead Ratio (GM) | **1.1097** |
| Benchmarks with SSA Overhead | 96/121 (79.3%) |

This means that **without optimization, SSA conversion adds ~11% more instructions** to the typical program due to phi lowering overhead.

#### 4.7.2 GVN Recovery

Our full GVN pipeline not only recovers from SSA overhead but achieves net optimization:

| Metric | Value |
|--------|-------|
| Total Instructions (After GVN) | 6,958 |
| **Net Change from Original** | **-1,131 instructions (-14.0%)** |
| GVN Final Ratio (GM) | **0.8403** |
| Benchmarks Improved by GVN | 60/121 (49.6%) |

The optimization pipeline must overcome **+891 instructions of SSA overhead** to achieve **-1,131 net savings**, meaning GVN actually eliminates **~2,022 instructions** worth of redundancy (891 + 1,131).

#### 4.7.3 Pipeline Stages Breakdown

For a typical benchmark, the instruction count changes through the pipeline:

```
Original Program
    ↓ +11.4% (SSA overhead)
After SSA Conversion + Lowering (no GVN)
    ↓ -24.5% (GVN optimization)
After Full GVN Pipeline
    = -16.0% net reduction from original
```

This demonstrates that:
1. **SSA is not free** — naive SSA round-trip adds significant overhead
2. **GVN more than compensates** — the optimization benefits outweigh SSA costs
3. **Copy coalescing is essential** — without it, SSA overhead would dominate

#### 4.7.4 Key Insight

The SSA overhead analysis reveals why our **copy coalescing** and **Phase 2 optimizations** (parameter renaming, constant copy fusion, phi destination propagation) are so important. Without these passes, the 11% SSA overhead would make many programs worse after "optimization."

---

## 5. Generative AI Usage

### 5.1 Tools Used

We used **Claude** and **ChatGPT** as primary AI assistants throughout this project. Claude was used via the command-line interface in the development environment, while ChatGPT was used through its web interface.

### 5.2 How We Used AI

1. **Debugging**: When encountering complex bugs (like the parallel copy infinite loop), we used Claude and ChatGPT to help trace through the algorithm and identify the root cause.

2. **Algorithm Clarification**: We asked for explanations of SSA concepts and their implementation details, particularly the difference between minimal and pruned SSA.

3. **Code Review**: After implementing each component, we used the AI tools to review the code for correctness and edge cases.

4. **Documentation**: The AI helped structure this report and ensure comprehensive coverage of all topics.

### 5.3 Illustrative Example

One particularly helpful interaction was debugging the parallel copy sequentialization. We presented the failing case:

```
Us: The parallel copy (x,y,z) := (y,z,x) causes an infinite loop
Claude: Let me trace through your algorithm...

The issue is that your cycle detection checks `if d in sources` but
in a 3-way cycle, ALL destinations are sources. The algorithm needs
to save one value to a temp to break the cycle.

Here's the fix:
1. Check if d is in BOTH sources AND dests (cycle member)
2. Save that value: tmp = d
3. Update all copies using d as source to use tmp instead
4. Now at least one copy becomes safe
```

This insight directly led to the fixed implementation.

### 5.4 When AI Was Unhelpful

The AI tools occasionally suggested overly complex solutions. For example, when we asked about copy elimination, they initially proposed a full interference graph construction with graph coloring — a solution appropriate for register allocation but overkill for copy propagation. The simpler dominator-tree walk approach was sufficient.

The tools also sometimes got confused about which phase of the pipeline we were in (SSA vs post-SSA), leading to suggestions that would have been unsound if implemented. This required careful verification on our part.

### 5.5 Assessment of AI Tools

**Strengths:**
- Excellent at explaining algorithms and their invariants
- Good at identifying edge cases and potential bugs
- Helpful for structuring complex code and documentation
- Fast iteration on debugging hypotheses

**Weaknesses:**
- Sometimes suggests overengineered solutions
- Can lose track of context in long debugging sessions
- Needs careful verification — doesn't always understand phase ordering
- May miss subtle correctness issues that require deep domain knowledge

**Conclusion**: AI tools are valuable assistants for compiler implementation, but they work best as a "pair programmer" where the human team maintains oversight and validates suggestions against the underlying theory. The combination of AI's pattern matching and human understanding of correctness invariants is powerful.

---

## 6. Conclusion

This project successfully implemented Global Value Numbering for Bril, achieving:

1. **100% Correctness**: All 123 benchmarks produce identical outputs after optimization
2. **Significant Optimization**: **15.9% static** and **18.3% dynamic** instruction reduction on average
3. **Broad Improvement**: 62/123 benchmarks (50.4%) show static improvement, with only 22/123 (17.9%) benchmarks worse in both metrics
4. **Algorithmic Fidelity**: Uses real phi nodes as described in the original GVN literature

The key technical contributions were:

- **Pruned SSA Implementation**: Reduced phi node bloat by 85%+, enabling effective optimization
- **Fixed Parallel Copy Sequentialization**: Correctly handles arbitrary copy patterns including cycles
- **SSA-Phase Copy Propagation**: Safely eliminates copies before phi lowering
- **Dominator-Tree Copy Elimination**: Efficiently removes remaining copies while respecting scope
- **Copy Coalescing**: Eliminates phi lowering overhead through liveness-based interference analysis
- **Phase 2 Optimizations**: Parameter renaming (with entry block safety), constant copy fusion, phi destination propagation, and TDCE function parameter handling

The hardest challenge was understanding why the optimizer was initially making code worse. Three key solutions transformed the project:
1. **Pruned SSA** — reduced unnecessary phi nodes by checking liveness at join points
2. **Copy Coalescing** — eliminated redundant copies introduced by phi lowering through interference analysis
3. **Phase 2 Optimizations** — further reduced SSA overhead by renaming parameters, fusing constants, and propagating phi destinations

Together, these techniques improved the static instruction ratio from 0.93x to **0.84x** and dynamic ratio from 0.92x to **0.81x**, achieving substantial code size and runtime reductions.

---

## 7. Comparison with Cooper/Briggs Paper {#7-comparison-with-cooperbriggs-paper}

This section compares our implementation against the [Cooper/Briggs 1997 paper](https://www.cs.tufts.edu/~nr/cs257/archive/keith-cooper/value-numbering.pdf) on value numbering.

### 7.1 Alignment with the Paper

Our implementation addresses all the core requirements of the DVNT (Dominator-based Value Numbering with Tables) algorithm:

| Issue | Status | Evidence |
|-------|--------|----------|
| **Real phi nodes** | **FIXED** | Uses `"op": "phi"` instructions. `to_ssa.py` creates real phis, `from_ssa.py` lowers them. |
| **DCE after GVN** | **FIXED** | `tdce.trivial_dce_func(func)` runs after GVN |
| **Dominator tree traversal** | **MATCHES PAPER** | `gvn_walk` does recursive dom-tree walk |
| **Scoped expression tables** | **MATCHES PAPER** | `expr2vn_stack` with `push_scope`/`pop_scope` |
| **Expression canonicalization** | **MATCHES PAPER** | Commutative sorting, relational flipping |
| **Meaningless phi detection** | **MATCHES PAPER** | Checks if all phi args have same VN |
| **Redundant phi detection** | **MATCHES PAPER** | Checks if phi pattern already in expression table |

### 7.2 Conservative Design Choices

| Aspect | Paper | Our Implementation | Impact |
|--------|-------|---------------------|--------|
| **Memory operations** | Suggests memory SSA or alias analysis | Conservative: clears expression table on effectful ops | Safe but limits optimization on `mem/` benchmarks |
| **Loop headers** | Can do iterative refinement | Conservative: fresh VN for backedge phis | Safe but may miss some loop-carried redundancies |

### 7.3 Features Not Implemented

The paper describes **two** algorithms:
- **DVNT** (Dominator-based Value Numbering with Tables) - hash-based, single pass
- **SCC-VN** (Strongly Connected Component Value Numbering) - partition-based, iterative

Our implementation is **DVNT only**. The following optional extensions are not implemented:

**1. SCC-Based Value Numbering (SCC-VN)**

SCC-VN can find more equivalences in loops with cyclic dependencies:

```
# Example where SCC-VN is better:
x.1 = phi(0, x.2)
y.1 = phi(0, y.2)
x.2 = y.1 + 1
y.2 = x.1 + 1
# SCC-VN can prove x.1 == y.1, DVNT cannot
```

**Impact**: Minor - DVNT is the more commonly implemented approach and handles most cases.

**2. Algebraic Simplifications**

The paper mentions that DVNT can be extended with algebraic identities:
- `x + 0 -> x`
- `x * 1 -> x`
- `x - x -> 0`
- `x AND true -> x`

**Impact**: Minor - these are typically done in a separate simplification pass.

**3. Global Code Motion**

The paper's "complete" algorithm includes **hoisting**: moving computations to dominators when beneficial. Our implementation only does **replacement** (substituting redundant computations with copies).

**Impact**: Minor for correctness, moderate for optimization power.

**4. Memory Versioning/SSA**

For loads/stores, the paper suggests treating memory as having "versions" so that:
```
store p, v1
x = load p      # x gets value v1
store p, v2
y = load p      # y gets value v2, not v1
```

Our implementation conservatively clears all expressions on any effectful operation.

**Impact**: Limits optimization on memory-heavy benchmarks.

### 7.4 Summary Comparison

| Feature | Paper (DVNT) | Our Implementation |
|---------|--------------|---------------------|
| Real SSA phi nodes | Required | Yes |
| Dom-tree traversal | Required | Yes |
| Scoped hash tables | Required | Yes |
| Meaningless phi removal | Yes | Yes |
| Redundant phi removal | Yes | Yes |
| Expression canonicalization | Yes | Yes |
| DCE post-pass | Recommended | Yes |
| SCC-VN (partition-based) | Optional | No |
| Algebraic simplifications | Optional | No |
| Global code motion | Optional | No |
| Memory SSA | Optional | No (conservative) |

### 7.5 Verdict

Our implementation faithfully implements the DVNT algorithm from the Cooper/Briggs paper. The critical structural fix (switching from get/set to real SSA phi nodes) addresses the core feedback, and we now operate exactly as the paper specifies for the required components.

The optional features we didn't implement (SCC-VN, algebraic simplifications, global code motion, memory SSA) are interesting extensions, but they're separate from the core DVNT algorithm. SCC-VN is a partition-based alternative to DVNT; algebraic simplifications and code motion are typical post-passes in production compilers but not part of the original algorithm; and memory SSA requires additional alias analysis infrastructure. Our conservative approach to memory operations is a documented trade-off: safe but limited on memory-heavy workloads.

In sum, we've implemented hash-based DVNT correctly and achieved meaningful optimization improvements. The report demonstrates that the algorithm works, and the engineering effort went into the practical details (SSA overhead elimination, copy coalescing) rather than into more speculative extensions.