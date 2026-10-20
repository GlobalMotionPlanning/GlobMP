"""Initial curves.  GlobMP restores feasibility from any start; a seed only selects which route class
(which side of each obstacle and robot) the result lies in.  Problem-specific seeds live with each
application: route-class enumeration in synthetic/, prioritized planning in applications/multi_robot."""
import torch

from .spline import DTYPE as DT


def init_waypoints(p, wp):
    """Least-squares chart point of the curve through M+1 waypoints wp (segment endpoints), linear inside."""
    wp = torch.as_tensor(wp, dtype=DT)             # (M+1,n)
    ss = torch.linspace(0, 1, p.d + 1, dtype=DT)
    tgt = torch.stack([wp[a].view(-1, 1) * (1 - ss) + wp[a + 1].view(-1, 1) * ss for a in range(p.M)], 0)  # (M,n,d+1)
    G = tgt.permute(1, 0, 2).reshape(p.n, -1)
    return torch.linalg.lstsq(p.N, (G - p.Gamma0).T).solution.T.contiguous()
