
## Base transit (standalone — not part of the live deploy)
`move_robot.py` drives the Vega mobile base a fixed distance (dexcontrol chassis, velocity ×
time, encoder-verified). It is a **manual utility** for multi-phase tasks (e.g. pick → move →
scan), run **between** policy phases — it is **not** invoked by `run_robot.sh` or the closed-loop
deploy. Keep the e-stop in hand.

```bash
python deploy/move_robot.py --distance 0.8      # forward 0.8 m
python deploy/move_robot.py --distance -0.8     # backward
```
