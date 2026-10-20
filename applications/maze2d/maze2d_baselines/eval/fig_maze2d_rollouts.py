"""Planned-vs-executed trajectory grids for maze2d, in the layout of DiRecT's
Figures 7 (broad) and 8 (narrow).

Rows are planning methods; columns are grouped by run id, each group holding the
diffused plan and the PD-tracked rollout.  Reads the `trajectories.npz` files
written by DiRecT's own `eval.py` (needs `eval.save_trajectories: true`), so the
cells are their arrays, not a re-render of their figure.

The drawing follows `Maze2dConstrainedEnv.render_maze` exactly, including its
`+0.7` observation offset, so a cell here is comparable to a cell there:
walls via `imshow(maze_arr == WALL)` on extent (0, 1, 1, 0); the trajectory as a
`jet`-coloured curve over its own length; keep-out regions filled, with the
enforced margin as an outline.

    python fig_maze2d_rollouts.py --setting broad --out figures/maze2d_broad_grid.pdf

A method with no data is drawn greyed and marked, so the grid can be filled in
as runs land.  `--ours` points at an npz supplying our row (keys
`planned_<run>` / `rollout_<run>`, each (T, 2) or (T, 4) in raw maze coords).
"""
import argparse, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

LARGE_MAZE = ["############", "#OOOO#OOOOO#", "#O##O#O#O#O#", "#OOOOOO#OOO#",
              "#O####O###O#", "#OO#O#OOOOO#", "##O#O#O#O###", "#OO#OOO#OGO#",
              "############"]
ISCALE, JSCALE, OFFSET = 9, 12, 0.7          # env.py MAZE_BOUNDS + render offset

# (center_x, center_y, radius_x, radius_y, power) -- env.py MAZE2D_*_OBSTACLES
OBSTACLES = {
    "broad": [(5.1, 4.3, 0.8, 0.8, 2), (4.6, 1.3, 0.8, 0.8, 4)],
    "narrow": [(4.8, 1.3, 1.0, 1.0, 4), (4.8, 4.3, 1.0, 1.0, 4),
               (2.3, 1.8, 1.0, 0.5, 4), (7.8, 2.8, 1.8, 1.8, 4),
               (6.9, 6.3, 1.0, 1.0, 4), (9.3, 5.6, 1.0, 1.0, 4)],
}
METHODS = [("no-guidance", "Diffuser"), ("classifier-guidance", "Classifier Guid."),
           ("gradient-guidance", "Gradient Guid."), ("projection", "Projected Diff."),
           ("primal-dual", "Primal-Dual"), ("augmented-lagrangian", "Aug. Lagrangian"),
           ("safediffuser-ros", "SafeDiff-RoS"), ("safediffuser-res", "SafeDiff-ReS"),
           ("safediffuser-tvs", "SafeDiff-TVS"), ("direct", "DiRecT"),
           ("__ours__", "GlobMP (ours)")]
WALLC, OBSFC, OBSEC, NAC = 0.5, "#d23b32", "#a81f17", "#ededed"


def maze_mask():
    return np.array([[1.0 if c == "#" else 0.0 for c in row] for row in LARGE_MAZE])


def to_plot(obs):
    """Raw maze observations -> render coordinates (x, y).

    Accepts (T, 2/4) observations and (T, 6) action-prefixed rows, which is how
    eval.py stacks actions ahead of the 4-D state; column 0 is y, column 1 is x
    (see compute_constraint_ellipses).  A leading singleton batch axis is
    squeezed, so a (1, T, d) array loads the same as (T, d).
    """
    obs = np.asarray(obs, dtype=float)
    while obs.ndim > 2 and obs.shape[0] == 1:
        obs = obs[0]
    if obs.ndim != 2:
        raise ValueError(f"expected a (T, d) trajectory, got {obs.shape}")
    if obs.shape[1] == 6:                 # actions prefixed to the state
        obs = obs[:, 2:]
    return (obs[:, 1] + OFFSET) / JSCALE, (obs[:, 0] + OFFSET) / ISCALE


def superellipse(cx, cy, rx, ry, p, margin=0.0):
    t = np.linspace(0, 2 * np.pi, 400)
    c, s = np.cos(t), np.sin(t)
    g = 1.0 + margin
    if p == 2:
        x, y = rx * g * c, ry * g * s
    else:                                     # env.py's own superellipse param
        x = rx * g * np.sqrt(np.abs(c)) * np.sign(c)
        y = ry * g * np.sqrt(np.abs(s)) * np.sign(s)
    return (x + cx + OFFSET) / JSCALE, (y + cy + OFFSET) / ISCALE


def draw_scene(ax, setting, margin):
    ax.imshow(maze_mask() * WALLC, extent=(0, 1, 1, 0), cmap=plt.cm.binary,
              vmin=0, vmax=1, interpolation="nearest", zorder=0)
    for (cx, cy, rx, ry, p) in OBSTACLES[setting]:
        x, y = superellipse(cx, cy, rx, ry, p)
        ax.fill(x, y, facecolor=OBSFC, edgecolor="none", alpha=0.42, zorder=3)
        if margin > 0:
            xm, ym = superellipse(cx, cy, rx, ry, p, margin)
            ax.plot(xm, ym, color=OBSEC, lw=0.4, ls=(0, (2.5, 1.5)), zorder=4)
        else:
            ax.plot(x, y, color=OBSEC, lw=0.4, zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.3); sp.set_color("0.6")


def draw_traj(ax, obs):
    x, y = to_plot(obs)
    pts = np.stack([x, y], 1).reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap=plt.cm.jet, linewidth=0.75, zorder=10,
                        capstyle="round")
    lc.set_array(np.linspace(0, 1, len(seg)))
    ax.add_collection(lc)
    ax.plot(x[0], y[0], "o", ms=1.9, mfc="none", mec="#101010", mew=0.5, zorder=12)
    ax.plot(x[-1], y[-1], "o", ms=1.9, color="#101010", mew=0, zorder=12)


def mark_missing(ax, text):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc=NAC, ec="none", zorder=6))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=3.4,
            color="0.42", style="italic", zorder=7)


def eval_dir(root, setting, policy):
    return os.path.join(root, "benchmark-dyn", setting, policy, "eval")


def load(root, setting, policy, run):
    f = os.path.join(eval_dir(root, setting, policy), str(run), "trajectories.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f)
    return d["generated_observations"], d["rollout_observations"]


def load_ours(path, run):
    if not path or not os.path.exists(path):
        return None
    d = np.load(path)
    kp, kr = f"planned_{run}", f"rollout_{run}"
    if kp not in d.files:
        return None
    return d[kp], (d[kr] if kr in d.files else None)


def read_margin(root, setting, policy):
    f = os.path.join(eval_dir(root, setting, policy), "config.yaml")
    if not os.path.exists(f):
        return 0.0
    for line in open(f):
        if "constraint_margin" in line:
            try:
                return float(line.split(":")[1].strip())
            except ValueError:
                return 0.0
    return 0.0


def main(a):
    for f in ("Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"):
        if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = f; break
    runs = a.runs
    methods = [m for m in METHODS if not (a.no_ours and m[0] == "__ours__")]
    ncol = 2 * len(runs)
    fig, axes = plt.subplots(len(methods), ncol,
                             figsize=(0.62 * ncol + 0.55, 0.47 * len(methods) + 0.42))
    if axes.ndim == 1:
        axes = axes[None, :]
    for r, (policy, name) in enumerate(methods):
        margin = 0.0 if policy == "__ours__" else read_margin(a.root, a.setting, policy)
        for k, run in enumerate(runs):
            data = (load_ours(a.ours, run) if policy == "__ours__"
                    else load(a.root, a.setting, policy, run))
            for half in (0, 1):
                ax = axes[r, 2 * k + half]
                draw_scene(ax, a.setting, margin)
                if data is None or data[half] is None:
                    mark_missing(ax, "pending" if policy == "__ours__" else "not run")
                else:
                    draw_traj(ax, data[half])
                if r == 0:
                    ax.set_title("Planned" if half == 0 else "Rollout",
                                 fontsize=4.0, pad=1.6)
        axes[r, 0].set_ylabel(name, rotation=90, ha="center", va="center",
                              fontsize=4.0, labelpad=2.4)
    fig.subplots_adjust(left=0.052, right=0.997, top=0.935, bottom=0.028,
                        wspace=0.05, hspace=0.05)
    # "Run ID: k" spanning each Planned|Rollout pair
    for k, run in enumerate(runs):
        p0 = axes[0, 2 * k].get_position(); p1 = axes[0, 2 * k + 1].get_position()
        fig.text(0.5 * (p0.x0 + p1.x1), 0.972, f"Run ID: {run}",
                 ha="center", va="bottom", fontsize=4.4)
    fig.legend(handles=[Patch(fc=OBSFC, ec=OBSEC, lw=0.4, alpha=0.42, label="keep-out region"),
                        Patch(fc=str(1 - WALLC), ec="none", label="maze wall")],
               loc="lower left", bbox_to_anchor=(0.052, 0.0005), ncol=2, frameon=False,
               fontsize=4.0, handlelength=1.1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=600)
    if a.png and a.out.endswith(".pdf"):
        fig.savefig(a.out[:-4] + ".png", dpi=600)
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["broad", "narrow"], default="broad")
    ap.add_argument("--root", default=os.path.join(
        os.environ.get("MB_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"),
        help="eval log root (<MB_ROOT>/runs, holding benchmark-dyn/<setting>/<policy>/eval)")
    ap.add_argument("--runs", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ours", default=None)
    ap.add_argument("--out", default="figures/maze2d_broad_grid.pdf")
    ap.add_argument("--png", action="store_true", help="also write a .png preview")
    ap.add_argument("--no-ours", action="store_true",
                    help="drop our row entirely instead of drawing it pending")
    main(ap.parse_args())
