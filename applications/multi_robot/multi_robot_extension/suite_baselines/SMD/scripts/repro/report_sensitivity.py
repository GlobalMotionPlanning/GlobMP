"""Sensitivity tables: each baseline's response curve along each difficulty axis."""
import argparse, glob, json, os, re
from collections import defaultdict

SET_ORDER = ['ctl', 'o11', 'o12', 'o13', 'o16', 'o17', 'o18', 'o19', 'o20', 'r060', 'r065']
SET_LABEL = {'ctl': 'obs 1.0x', 'o11': 'obs 1.1x', 'o12': 'obs 1.2x',
             'o13': 'obs 1.3x', 'o16': 'obs 1.6x', 'o17': 'obs 1.7x', 'o18': 'obs 1.8x',
             'o19': 'obs 1.9x', 'o20': 'obs 2.0x', 'r060': 'r=0.060', 'r065': 'r=0.065'}
AXIS = {'r060': 'radius', 'r065': 'radius',
        'o11': 'obstacle', 'o12': 'obstacle', 'o13': 'obstacle', 'o16': 'obstacle',
        'o17': 'obstacle', 'o18': 'obstacle', 'o19': 'obstacle', 'o20': 'obstacle'}


def load(out):
    rows = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(out, '*_*.json'))):
        m = re.match(r'(smd|direct)_(.+)_(' + '|'.join(SET_ORDER) + r')\.json$',
                     os.path.basename(f))
        if not m:
            continue
        meth, tag, sname = m.groups()
        try:
            d = json.load(open(f))
        except Exception:
            continue
        rows[(meth, tag)][sname] = d
    return rows


def fmt(d):
    if d is None:
        return '   -  '
    s = d.get('success_rate')
    return '  --  ' if s is None else f'{s:5.2f} '


def main(out):
    rows = load(out)
    if not rows:
        print('  no results yet'); return
    for meth in ('smd', 'direct'):
        keys = sorted([k for k in rows if k[0] == meth],
                      key=lambda k: (k[1].split('-')[0], int(k[1].split('-')[1])))
        if not keys:
            continue
        print(f'\n=== {meth.upper()}  success rate (10 instances/cell, test-only) ===')
        print(f"  {'cell':<14}" + ''.join(f'{SET_LABEL[s]:>10}' for s in SET_ORDER)
              + f"{'drop_rad':>10}{'drop_obs':>10}")
        for k in keys:
            got = rows[k]
            line = f'  {k[1]:<14}'
            for s in SET_ORDER:
                line += fmt(got.get(s)).rjust(10)
            c = (got.get('ctl') or {}).get('success_rate')
            def drop(names):
                vals = [(got.get(n) or {}).get('success_rate') for n in names]
                vals = [v for v in vals if v is not None]
                if c is None or not vals:
                    return '     -    '
                return f'{min(vals) - c:+9.2f} '
            line += drop(['r060', 'r065']) + drop(['o11', 'o12', 'o13', 'o16', 'o17', 'o18', 'o19', 'o20'])
            print(line)
        # incompleteness is reported, never silently averaged over
        part = [(k[1], s, d.get('n_instances'))
                for k, v in rows.items() if k[0] == meth
                for s, d in v.items() if d.get('n_instances', 10) < 10]
        if part:
            print('  [incomplete] ' + ', '.join(f'{t}/{s}:{n}/10' for t, s, n in part))
    print('\n  drop_* = worst success on that axis minus control. '
          'Negative = the axis degrades the method.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    main(ap.parse_args().out)
