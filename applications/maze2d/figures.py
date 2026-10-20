"""The two Maze2D figures of the paper, drawn from the npz files plan.py writes.

    maze2d_ours_paths       the continuous plan, every 16th of the H = 384 waypoints the benchmark receives,
                            and the rollout executed by the benchmark's PD controller, on both settings, with
                            a magnified view of the rollout's tightest pass
    maze2d_ours_clearance   keep-out clearance min_j level_j - (1 + margin) over the episode

    python figures.py --broad results/broad.npz --narrow results/narrow.npz --out-dir figures

Each npz holds dense (S, 2), waypoints (H, 4), rollout (steps+1, 4), rects (M, 2, 2), obstacles (K, 5) as
(cy, cx, ry, rx, p), start, goal and margin; rows are maze coordinates (y, x).  The figures need only
numpy and matplotlib, not the benchmark.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# print style of the paper: drawn at the ICLR text width so fonts print at their nominal size, Times if installed
for _f in glob.glob('/usr/share/fonts/truetype/msttcorefonts/[Tt]imes*.ttf'):
    font_manager.fontManager.addfont(_f)
TEXTWIDTH = 5.5
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, GRID = '#222222', '#5a5a5a', '#e9e9e6'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'STIXGeneral'], 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
    'lines.linewidth': 1.1, 'lines.markersize': 3.8, 'errorbar.capsize': 1.6,
    'legend.frameon': False, 'legend.handlelength': 2.2, 'legend.columnspacing': 1.2, 'legend.handletextpad': 0.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

# maze2d-large (d4rl LARGE_MAZE): '#' wall, 'O' free, 'G' goal; cell (i, j) is centred at (y, x) = (i, j)
LARGE_MAZE = ["############", "#OOOO#OOOOO#", "#O##O#O#O#O#", "#OOOOOO#OOO#", "#O####O###O#",
              "#OO#O#OOOOO#", "##O#O#O#O###", "#OO#OOO#OGO#", "############"]
KO_FC, KO_EC, WALL, BAND = '#d23b32', '#a81f17', '#9b9b98', '#dbe8f7'


def level(P, ob):
    """Benchmark keep-out value |dy/ry|^p + |dx/rx|^p; a point is outside the keep-out iff it is >= 1."""
    cy, cx, ry, rx, p = ob
    return np.abs((P[:, 0] - cy) / ry) ** p + np.abs((P[:, 1] - cx) / rx) ** p


def draw_maze(ax, d):
    for i, row in enumerate(LARGE_MAZE):
        for j, c in enumerate(row):
            if c == '#': ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=WALL, ec='none', zorder=1))
    for lo, hi in d['rects']:
        ax.add_patch(Rectangle((lo[1], lo[0]), hi[1] - lo[1], hi[0] - lo[0], fc=BAND, ec='none', alpha=0.55, zorder=0))
    ys, xs = np.mgrid[-0.5:8.5:600j, -0.5:11.5:800j]; G = np.stack([ys.ravel(), xs.ravel()], 1)
    m = float(d['margin'])
    for ob in d['obstacles']:
        L = level(G, ob).reshape(ys.shape)
        ax.contourf(xs, ys, L, levels=[0, 1], colors=[KO_FC], alpha=0.85, zorder=2)
        ax.contour(xs, ys, L, levels=[1 + m], colors=[KO_EC], linewidths=0.6, linestyles=[(0, (3, 1.6))], zorder=2)


def paths(ax, ins, d, title):
    def layer(a):
        a.plot(d['dense'][:, 1], d['dense'][:, 0], color=BLUE, lw=2.2, zorder=4, solid_capstyle='round')
        w = d['waypoints'][::16]
        a.plot(w[:, 1], w[:, 0], 'o', ms=1.5, color=INK, zorder=6)
        a.plot(d['rollout'][:, 1], d['rollout'][:, 0], color=ORANGE, lw=0.9, ls=(0, (3, 1.2)), zorder=5)
    draw_maze(ax, d); layer(ax)
    s, g = d['start'], d['goal']
    ax.plot(s[1], s[0], 'o', ms=5, mfc='white', mec=INK, mew=1.0, zorder=7)
    ax.plot(g[1], g[0], '*', ms=8, color=INK, zorder=7)
    ax.set_xlim(-0.5, 11.5); ax.set_ylim(8.5, -0.5); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=8, color=INK, pad=2)
    # magnify the executed rollout's tightest pass (where tracking error goes deepest into the margin)
    lv = np.min([level(d['rollout'][:, :2], ob) for ob in d['obstacles']], 0)
    k = int(np.argmin(lv)); cy, cx = d['rollout'][k, :2]; r = 0.55
    draw_maze(ins, d); layer(ins)
    ins.set_xlim(cx - r, cx + r); ins.set_ylim(cy + r, cy - r); ins.set_aspect('equal')
    ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values(): sp.set_edgecolor(INK2); sp.set_linewidth(0.6); sp.set_visible(True)
    ins.set_title('tightest executed pass', fontsize=6.4, color=INK2, pad=2)
    ax.indicate_inset_zoom(ins, edgecolor=INK2, lw=0.6, alpha=0.9)


def save(fig, stem):
    fig.savefig(stem + '.pdf'); fig.savefig(stem + '.png'); plt.close(fig)


def figure_paths(D, out):
    fig = plt.figure(figsize=(TEXTWIDTH, 1.95))
    gs = fig.add_gridspec(1, 4, width_ratios=[3.0, 1.15, 3.0, 1.15], left=0.005, right=0.995, top=0.9, bottom=0.17, wspace=0.08)
    A = [fig.add_subplot(gs[0, i]) for i in range(4)]
    paths(A[0], A[1], D['broad'], '(a) Broad, two keep-outs')
    paths(A[2], A[3], D['narrow'], '(b) Narrow, six keep-outs')
    h = [Line2D([], [], color=BLUE, lw=2.2), Line2D([], [], ls='', marker='o', ms=2.4, color=INK),
         Line2D([], [], color=ORANGE, lw=0.8, ls=(0, (3, 1.2))),
         Rectangle((0, 0), 1, 1, fc=KO_FC, ec='none'), Line2D([], [], color=KO_EC, lw=0.6, ls=(0, (3, 1.6))),
         Rectangle((0, 0), 1, 1, fc=BAND, ec='none')]
    fig.legend(h, ['GlobMP plan (continuous)', 'waypoints (every 16th)', 'executed rollout', 'keep-out',
                   'enforced margin', 'corridor'], loc='lower center', ncol=6, frameon=False, fontsize=6.4,
               handlelength=1.8, columnspacing=0.9, bbox_to_anchor=(0.5, 0.0))
    save(fig, os.path.join(out, 'maze2d_ours_paths'))


def figure_clearance(D, out):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.9), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.3, wspace=0.08)
    for ax, (s, title) in zip(axes, [('broad', '(a) Broad'), ('narrow', '(b) Narrow')]):
        d = D[s]; m = float(d['margin'])
        clr = lambda P: np.min([level(P, ob) for ob in d['obstacles']], 0) - (1 + m)
        H = len(d['waypoints'])                                   # the PD controller tracks waypoint min(step, H-1)
        td = np.linspace(0, H - 1, len(d['dense'])); tw = np.arange(H); tr = np.arange(len(d['rollout']))
        ax.axhspan(-(1 + m), -m, color=KO_FC, alpha=0.18, lw=0, zorder=0)           # inside the keep-out itself
        ax.axhline(0, color=KO_EC, lw=0.7, ls=(0, (3, 1.6)), zorder=1)              # the enforced margin
        ax.plot(td, clr(d['dense']), color=BLUE, lw=1.4, zorder=3)
        ax.plot(tw[::6], clr(d['waypoints'][::6, :2]), 'o', ms=1.6, color=INK, zorder=4)
        ax.plot(tr, clr(d['rollout'][:, :2]), color=ORANGE, lw=0.8, ls=(0, (3, 1.2)), zorder=5)
        ax.set_ylim(-0.75, 3.0); ax.set_xlim(0, H + 20)
        ax.grid(axis='y', color=GRID, lw=0.5)
        ax.set_title(title, fontsize=8, color=INK, pad=2); ax.set_xlabel('environment step', fontsize=7.5)
        print(f"{s}: plan min clearance {clr(d['dense']).min():.3e}, rollout min clearance "
              f"{clr(d['rollout'][:, :2]).min():.4f}, rollout min level "
              f"{np.min([level(d['rollout'][:, :2], ob) for ob in d['obstacles']]):.4f} (>= 1 is outside)")
    axes[0].set_ylabel('keep-out clearance', fontsize=7.5)
    h = [Line2D([], [], color=BLUE, lw=1.4), Line2D([], [], ls='', marker='o', ms=2.4, color=INK),
         Line2D([], [], color=ORANGE, lw=0.9, ls=(0, (4, 1.4))), Line2D([], [], color=KO_EC, lw=0.7, ls=(0, (3, 1.6))),
         Rectangle((0, 0), 1, 1, fc=KO_FC, alpha=0.18, ec='none')]
    fig.legend(h, ['GlobMP plan (continuous)', 'benchmark waypoints', 'executed rollout', 'enforced margin',
                   'inside keep-out'], loc='lower center', ncol=5, frameon=False, fontsize=6.4, handlelength=1.8,
               columnspacing=1.0, bbox_to_anchor=(0.54, 0.0))
    save(fig, os.path.join(out, 'maze2d_ours_clearance'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--broad', default=os.path.join('results', 'broad.npz'))
    ap.add_argument('--narrow', default=os.path.join('results', 'narrow.npz'))
    ap.add_argument('--out-dir', default='figures')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    D = {'broad': np.load(a.broad), 'narrow': np.load(a.narrow)}
    figure_paths(D, a.out_dir)
    figure_clearance(D, a.out_dir)
    print(f"wrote {a.out_dir}/maze2d_ours_paths.{{pdf,png}} and {a.out_dir}/maze2d_ours_clearance.{{pdf,png}}")


if __name__ == '__main__':
    main()
