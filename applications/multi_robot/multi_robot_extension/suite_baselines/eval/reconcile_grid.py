#!/usr/bin/env python3
"""Rebuild grid state from the result files the runners themselves write.

Needed because the state file has two writers whenever anything is adjusted by hand
while the dispatcher runs: read-modify-write from outside loses every transition the
dispatcher made in between. That is not cosmetic -- a cell left marked 'pending' while
its process is still alive gets dispatched a SECOND time, and two processes then write
one output file.

Status here is derived, not remembered: a cell is done if its result file exists, is
non-empty, and was written during this sweep; otherwise pending. 'blocked' is preserved,
and cells whose process is alive right now are marked running so they are not re-issued.

    ./reconcile_grid.py --since '2026-09-17 00:00:00' [--apply]

Without --apply it only reports. Stop the dispatcher before applying.
"""
import argparse, json, os, re, subprocess, sys, time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # suite_baselines/
OUT = os.environ['GRID_OUT']


def _result_paths(v):
    """Where this method leaves its per-cell result.

    Eight of the nine write the --timing-json path grid.py hands them. diffusion_policy
    goes through run_dp.sh, which has no such flag: inference_pcdiff writes its own tree,
    and the run tag (grid.py passes the velocity regime) is what separates the three
    regimes of one (map, N). Treating its absence from grid_out as "not run" would have
    requeued 121 finished cells.
    """
    if v['method'] == 'diffusion_policy':
        import glob
        pat = os.path.join(os.environ['DP_WORK'], 'results', f"dp_{v['map']}",
                           f"agent___{v['n']}", 'method___*', 'cost___*',
                           f"*-dp_{v['map']}_a{v['n']}_i*_s*_{v['regime']}", '*', 'results.json')
        return sorted(glob.glob(pat))
    return [os.path.join(os.environ['GRID_OUT'],
                         f"{v['method']}__{v['map']}__N{v['n']}__{v['regime']}.json")]


def _mtime(v):
    return max((os.path.getmtime(p) for p in _result_paths(v) if os.path.exists(p)),
               default=0.0)


def _has_result(v, cutoff):
    return any(os.path.exists(p) and os.path.getsize(p) > 0 and os.path.getmtime(p) >= cutoff
               for p in _result_paths(v))


def live_cells():
    """(method, map, n, regime) for every cell process running right now.

    The output path each cell was given carries its identity, so one regex over the
    command lines identifies them without trusting the state file.
    """
    try:
        ps = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout
    except Exception:
        return set()
    out = set()
    for line in ps.splitlines():
        m = re.search(r'/grid_out/([a-z_]+)__([a-z]+)__N(\d+)__(low|medium|high)\.json', line)
        if m:
            out.add(f'{m.group(1)}/{m.group(2)}/N{m.group(3)}/{m.group(4)}')
        else:                      # diffusion_policy goes through run_dp.sh: positional args
            m = re.search(r'run_dp\.sh\s+(\w+)\s+(\d+)\s+\d+\s+\d+\s+\S+\s+\S+\s+(low|medium|high)', line)
            if m:
                out.add(f'diffusion_policy/{m.group(1)}/N{m.group(2)}/{m.group(3)}')
    out |= _live_smd()
    return out


def _live_smd():
    """SMD cells, read from each process's own environment.

    SMD goes through run_scenario1_pilot.sh and is given no --timing-json, so its command
    line carries no grid_out path and the regex above cannot see it. A cell invisible here
    is reported as not-running and gets dispatched a SECOND time: that happened, and two
    evaluate_smd_C.py emptyhard-3 processes wrote the same model directory at once. The
    tag gives map and team size; the regime is only in SMD_VEL_REGIME.
    """
    import glob
    found = set()
    for c in glob.glob('/proc/[0-9]*/cmdline'):
        try:
            parts = [x for x in open(c, 'rb').read().split(b'\0') if x]
            # TAG, not the command line. Matching evaluate_smd_C.py only sees a cell in
            # its EVALUATION stage; a cell still in stages 1-5 (data generation, training,
            # PP, merge) runs under other names and was invisible -- dense was mid
            # generate_via_pp and reported as not running. Those are exactly the stages
            # that WRITE the map's shared artifacts, so a second cell dispatched onto the
            # same map corrupts them. run_scenario1_pilot.sh exports TAG for every stage.
            pid = c.split('/')[2]
            try:
                env = dict(l.split('=', 1) for l in
                           open('/proc/%s/environ' % pid).read().split('\0') if '=' in l)
            except Exception:
                continue
            tag = env.get('TAG', '')
            m = re.match(r'([a-z]+)hard-(\d+)$', tag)
            if not m or not env.get('SMD_BASE'):
                continue
            found.add('smd/%s/N%s/%s' % (m.group(1), m.group(2),
                                         env.get('SMD_VEL_REGIME', 'medium')))
        except Exception:
            continue
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', default=os.path.join(ROOT, 'grid_state_shared.json'))
    ap.add_argument('--since', required=True,
                    help="only result files written after this count as done -- earlier runs "
                         "wrote to the same directory")
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    cutoff = datetime.strptime(a.since, '%Y-%m-%d %H:%M:%S').timestamp()
    st = json.load(open(a.state))
    alive = live_cells()
    tally = {}
    for k, v in st['cells'].items():
        if v['status'] == 'blocked':
            tally['blocked'] = tally.get('blocked', 0) + 1
            continue
        has = _has_result(v, cutoff)
        if k in alive:
            new = 'running'
        elif has:
            new = 'done'
        else:
            new = 'pending'
        tally[new] = tally.get(new, 0) + 1
        if new != v['status']:
            tally.setdefault('_changed', []).append(f"{k}: {v['status']} -> {new}")
        if new == 'done':
            # A cell with no `started` has no run window, and score_std's pick() refuses
            # to match a saved run directory without one -- it would otherwise match ANY
            # run in the pool. Cells promoted from another column (the old Low became the
            # new High when the triple moved) arrive here with started=None, and 63 of
            # them silently scored as empty. Derive a window from the result file.
            if not v.get('started'):
                _m = _mtime(v)
                if _m:
                    v['started'] = time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(_m - 3600))
        if new == 'done':
            v.update(status='done', rc=0,
                     error=v.get('error') or 'status derived from result file',
                     ended=v.get('ended') or time.strftime('%Y-%m-%d %H:%M:%S',
                                                           time.localtime(_mtime(v))))
        elif new == 'running':
            v['status'] = 'running'
        else:
            v.update(status='pending', rc=None, error=None, started=None, ended=None, secs=None)
    ch = tally.pop('_changed', [])
    print({k: tally[k] for k in sorted(tally)})
    print(f'changes: {len(ch)}')
    for c in ch[:15]:
        print('  ' + c)
    if len(ch) > 15:
        print(f'  ... and {len(ch) - 15} more')
    if a.apply:
        json.dump(st, open(a.state, 'w'), indent=1)
        print(f'applied to {a.state}')
    else:
        print('(dry run -- pass --apply)')


if __name__ == '__main__':
    main()
