#!/usr/bin/env python3
"""gate.py <checks_dir>: PASS/FAIL from h5.json + videos.txt. A task that fails is never converted.

Allowed: all-zero tactile force on a hand (the idle hand of a one-handed task; the sensor is live).
Fatal: any other h5 problem (including result.json success != true), a video that does not fully
decode, has decode errors, is not 640x360, or whose frame count differs from the h5 rows by > 1."""
import json, sys
D = sys.argv[1]; s = json.load(open(f"{D}/h5.json")); h5 = {r["episode"]: r["N"] for r in s["episodes"]}
probs = [p for p in s["summary"]["problems"] if "tactile force identically zero" not in p[1]]
rows = [l.split() for l in open(f"{D}/videos.txt") if l.strip()]
vbad = [r for r in rows if int(r[2]) < 0 or abs(int(r[2]) - h5.get(r[0], -9)) > 1 or r[3] != "640x360" or int(r[5]) > 0]
ok = not probs and not vbad and len(rows) == 4 * s["summary"]["episodes"]
print(("PASS" if ok else "FAIL") + f" eps={s['summary']['episodes']} h5-problems={len(probs)} videos={len(rows)} video-problems={len(vbad)}")
for p in probs[:20]: print("  h5:", *p)
for r in vbad[:20]: print("  video:", *r)
sys.exit(0 if ok else 1)
