"""Replace the baked-in panel titles of figures/maze2d_maps_v2.png.

The figure was produced by a coauthor and no generator exists in the repo, so the
plotted content cannot be rebuilt from data. This script edits only the title band
(the blank rows above both panels): every plotted pixel is copied through verbatim.

Old titles carried evaluative wording ("certified" / "not certified") that became
misleading once the Narrow row reported SR 1.000, Viol. 0.000; the certification
distinction now lives in the LaTeX caption instead. Both titles are left-aligned to
their panel's left edge (the original centred (b); (a) was already left-aligned).

Usage:  python fig_maze2d_retitle.py [--check]
"""
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "figures/maze2d_maps_v2.png"

BAND_H = 56          # blank rows above the panels; panel content starts at row 60
BASELINE = 43        # baseline of the original title text
PANEL_LEFT = (22, 1138)   # left edge of each panel's title, in pixels
TITLES = (r"(a) Broad — disc-cover formulation",
          r"(b) Narrow — exact $p = 4$ formulation")
CALIB = r"(a) Broad — certified trajectory"   # the original (a) title...
TARGET_W = 482                                # ...and its measured ink width
DPI = 200


def render_strip(width, fontsize, titles=TITLES):
    """Title text on a transparent strip, `width` x BAND_H pixels."""
    fig = plt.figure(figsize=(width / DPI, BAND_H / DPI), dpi=DPI)
    fig.patch.set_alpha(0.0)
    for x, t in zip(PANEL_LEFT, titles):
        fig.text(x / width, 1.0 - BASELINE / BAND_H, t, fontsize=fontsize,
                 ha="left", va="baseline", color="#000000")
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def ink_width(rgba, lo, hi):
    """Horizontal extent of non-transparent pixels within columns [lo, hi)."""
    cols = np.where(rgba[:, lo:hi, 3].max(0) > 8)[0]
    return int(cols[-1] - cols[0] + 1) if len(cols) else 0


def main():
    base = np.asarray(Image.open(SRC).convert("RGB")).copy()
    h, w, _ = base.shape

    # Recover the original title's optical size: pick the font size at which the
    # ORIGINAL (a) string renders at its measured ink width, then reuse that size.
    # Ink width is a step function of font size (hinting), so scan rather than bisect.
    grid = [8.0 + 0.05 * i for i in range(161)]
    fontsize = min(grid, key=lambda f: abs(
        ink_width(render_strip(w, f, (CALIB, "")), 0, PANEL_LEFT[1] - 40) - TARGET_W))
    print(f"font size {fontsize:.2f} pt reproduces the original title at "
          f"{ink_width(render_strip(w, fontsize, (CALIB, '')), 0, PANEL_LEFT[1] - 40)} px "
          f"(measured {TARGET_W})")
    strip = render_strip(w, fontsize)
    print(f"new (a) ink width {ink_width(strip, 0, PANEL_LEFT[1] - 40)} px, "
          f"new (b) ink width {ink_width(strip, PANEL_LEFT[1] - 40, w)} px")

    if strip[:, :, 3].max() and np.where(strip[:, :, 3].max(1) > 8)[0].max() >= BAND_H:
        sys.exit("title text would collide with the panels")

    if "--check" in sys.argv:
        return

    out = base.copy()
    out[:BAND_H] = 255                      # clear the old titles
    a = strip[:, :, 3:4].astype(np.float32) / 255.0
    out[:BAND_H] = (strip[:, :, :3] * a + out[:BAND_H] * (1 - a)).round().astype(np.uint8)

    assert (out[BAND_H:] == base[BAND_H:]).all(), "plotted content was modified"
    Image.fromarray(out).save(SRC)
    print(f"wrote {SRC}")


if __name__ == "__main__":
    main()
