#!/usr/bin/env python3
"""Assign (GPU, cpuset, threads) per cell for this box.

GPU0-3 hang off NUMA node 0 (cpus 0-31,64-95); GPU4-7 off node 1 (32-63,96-127)
-- per `nvidia-smi topo -m`. A job wants the CPUs on its GPU's own node, or every
host->device copy crosses the socket interconnect.

Free GPUs are detected at call time, because GPUs 0,1,2,6 belong to another user's
job and that can change. Anything busy is left strictly alone.
"""
import subprocess, sys, json, argparse

NODE_CPUS = {0: list(range(0, 32)) + list(range(64, 96)),
             1: list(range(32, 64)) + list(range(96, 128))}
GPU_NODE = {g: (0 if g < 4 else 1) for g in range(8)}
RESERVE_PER_NODE = 8          # headroom for the other tenant + system

def free_gpus(mem_mib=500, util=15, need_free_mib=8000):
    """Rank GPUs by how usable they are, rather than demanding an idle one.

    This box is shared: another user (another user) can claim all 8 cards within minutes,
    as happened mid-run. Requiring util<15 then yields "no free GPUs" and refuses to
    launch, even though a 40 GB A100 running someone's 2.8 GB job has ample room and
    our small 2D UNet barely competes for SMs. So: exclude only cards without
    need_free_mib headroom, and order the rest least-loaded-first."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    cand, full = [], []
    for line in out.strip().splitlines():
        i, m, tot, u = [int(x.strip()) for x in line.split(",")]
        (cand if (tot - m) >= need_free_mib else full).append((i, m, u))
    cand.sort(key=lambda t: (t[2], t[1]))          # utilisation, then memory used
    idle = [i for i, m, u in cand if m < mem_mib and u < util]
    return [i for i, _, _ in cand], [i for i, _, _ in full], idle

def _ranges(cpus):
    """Collapse a cpu list into ranges. Each NUMA node owns two disjoint blocks
    (0-31,64-95), so a contiguous slice is genuinely two ranges -- printing
    first-last would read as one span crossing the other socket."""
    cs = sorted(int(c) for c in cpus); out = []
    for c in cs:
        if out and c == out[-1][1] + 1: out[-1][1] = c
        else: out.append([c, c])
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in out)


def plan(tags, concurrency=None, gpus=None):
    avail, busy, idle = free_gpus()
    if gpus: avail = [g for g in gpus if g in avail]
    if not avail:
        raise SystemExit(f"no GPU has headroom (all near-full: {busy})")
    n = concurrency or len(avail)
    slots = [(i, avail[i % len(avail)]) for i in range(n)]
    per_node = {}
    for _, g in slots: per_node.setdefault(GPU_NODE[g], []).append(g)
    cursor = {node: RESERVE_PER_NODE for node in per_node}
    thr = {node: max(1, (len(NODE_CPUS[node]) - RESERVE_PER_NODE) // len(gs))
           for node, gs in per_node.items()}
    out = []
    for slot, g in slots:
        node = GPU_NODE[g]; t = thr[node]
        cpus = NODE_CPUS[node][cursor[node]:cursor[node] + t]; cursor[node] += t
        out.append({"slot": slot, "gpu": g, "node": node, "threads": len(cpus),
                    "cpuset": ",".join(map(str, cpus)),
                    "tags": tags[slot::n] if tags else []})
    return out, avail, busy, idle

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*"); ap.add_argument("-j", "--concurrency", type=int)
    ap.add_argument("--gpus", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    p, avail, busy, idle = plan(a.tags, a.concurrency, a.gpus)
    if a.json: print(json.dumps(p)); sys.exit()
    print(f"usable (least-loaded first): {avail}\n"
          f"fully idle: {idle or 'none -- box is contended'}    no headroom: {busy or 'none'}")
    for s in p:
        print(f"  slot {s['slot']}: GPU {s['gpu']} (NUMA {s['node']})  "
              f"{s['threads']:>2} threads  cpus {_ranges(s['cpuset'].split(','))}"
              f"  cells: {' '.join(s['tags']) or '-'}")
