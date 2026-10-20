"""Algorithm 2: gradient projection.

From a feasible curve, take gradient steps on the cost (with block momentum) and restore feasibility
with Algorithm 1; a step is accepted only if the restored curve is feasible and cheaper, so every
accepted iterate is a complete feasible trajectory and the cost decreases monotonically."""
import time



class GPAMixin:
    def solve(self, Z, tv, blocks=600, steps=5, eta=None, deadline=None, verbose=False):
        """Restore from Z, then run up to `blocks` outer steps.  Returns (Z, step size, feasible).
        self.hist records (cost, seconds since the start of Algorithm 2) per accepted step."""
        Z, eta, ok = self.project(Z, tv, eta, deadline=deadline)
        if not ok: return Z, eta, False
        Zp = Z.clone(); tc = 0; hist = []; t_start = time.time(); self.hist = []
        for blk in range(blocks):
            if deadline and time.time() > deadline: break
            tc += 1; y = Z + ((tc - 1) / (tc + 2)) * (Z - Zp)
            for _ in range(steps): y = (y - (1 / self.beta) * self.cost_grad(y)).detach()
            cand, eta, okp = self.project(y, tv, eta, cap=600, deadline=deadline)
            if not okp or float(self.cost(cand)) > float(self.cost(Z)): tc = 0; Zp = Z.clone(); continue
            Zp = Z; Z = cand; hist.append(float(self.cost(Z)))
            self.hist.append((hist[-1], time.time() - t_start))
            if verbose and blk % 50 == 0: print(f"   blk {blk} cost {hist[-1]:.4f}", flush=True)
            if len(hist) > 60 and hist[-51] - hist[-1] < self.stop_rel * abs(hist[-1]): break
        return Z, eta, True
