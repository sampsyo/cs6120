import json
import sys

TERMINATORS = {"jmp", "br", "ret"}

def form_blocks(instrs):
    """
    Split instructions into basic blocks.
    A new block starts at:
      - the first instruction,
      - any labeled instruction,
      - the instruction after a terminator (jmp/br/ret).
    """
    blocks, cur, label2block = [], [], {}
    def flush():
        nonlocal cur
        if not cur:
            return
        first = cur[0]
        name = first.get("label") if "label" in first else f"@B{len(blocks)}"
        blocks.append({"name": name, "instrs": cur})
        if "label" in first:
            label2block[first["label"]] = name
        cur = []

    for ins in instrs:
        if "label" in ins and cur:
            flush()
        cur.append(ins)
        if ins.get("op") in TERMINATORS:
            flush()
    flush()
    return blocks, label2block

def build_cfg(blocks, label2block):
    """
    Build successors for each block.
    """
    succ = {b["name"]: [] for b in blocks}
    for i, b in enumerate(blocks):
        if not b["instrs"]:
            continue
        last = b["instrs"][-1]
        op = last.get("op")
        if op == "jmp":
            target = last["labels"][0]
            succ[b["name"]].append(label2block.get(target, target))
        elif op == "br":
            t, f = last["labels"][:2]
            succ[b["name"]].append(label2block.get(t, t))
            succ[b["name"]].append(label2block.get(f, f))
        elif op == "ret":
            pass  # no successors
        else:
            if i + 1 < len(blocks):
                succ[b["name"]].append(blocks[i + 1]["name"])
    return succ

def mycfg():
    prog = json.load(sys.stdin)
    for func in prog.get("functions", []):
        blocks, label2block = form_blocks(func.get("instrs", []))
        succ = build_cfg(blocks, label2block)

        print(f"Function {func['name']}:")
        print("  Blocks:", ", ".join(b["name"] for b in blocks))
        print("  Entry:", blocks[0]["name"] if blocks else "(none)")
        print("  Edges:")
        for b in blocks:
            outs = succ[b["name"]]
            if outs:
                print(f"    {b['name']} -> {', '.join(outs)}")
            else:
                print(f"    {b['name']} -> (exit)")
        print()

if __name__ == "__main__":
    mycfg()
