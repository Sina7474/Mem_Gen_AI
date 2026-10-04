"""
replot_28GHz_LoS.py — Redraw beam_alignment_N{ddim_n}.png|pdf from cached results
=================================================================================
Reuses the SNR values already written by run_28GHz_LoS.py
(28GHz_LoS/results/checkpoints/*.json). No model is trained and no synthetic
pool is loaded; only the figure is redrawn.

Usage:
    conda activate Mem_Gen
    cd Downstream_Tasks/Beam_alignment
    python replot_28GHz_LoS.py --ddim_n 100 1000
"""

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

import run_28GHz_LoS as R
from data_utils_28GHz_LoS import coord_keys, load_ddim_train_coords
from beam_align_core import compute_baselines


def cached_snr(name, npb):
    f = R.CKPT_DIR / f"{name}_npb{npb}.json"
    if not f.exists():
        raise FileNotFoundError(f"Missing cached metric: {f}")
    return json.loads(f.read_text())["snr"]


def _tau_label(tau):
    """Short tau label for the compact figure: 10000 -> '10k'."""
    return f"{tau // 1000}k" if tau % 1000 == 0 else str(tau)


def save_legend_only(handles, labels, stem, ncol=1, fontsize=16,
                     width_in=None):
    """Write the legend on its own canvas.

    ``width_in`` forces the saved canvas to exactly that width (inches) with the
    legend centred, so that including it at ``\linewidth`` reproduces the same
    scale factor -- and therefore the same rendered point size -- as the other
    figures. When ``None`` the canvas is cropped tightly to the legend box.
    """
    blank = Line2D([], [], linestyle="none", marker="none")
    if ncol == 2 and len(handles) == 5:
        # 2 columns x 3 rows, filled column-major: MRT / Genie / Real Only in
        # the first column, the two tau curves in the second.
        order = [0, 1, 2, 3, 4, None]
        handles = [blank if i is None else handles[i] for i in order]
        labels = ["" if i is None else labels[i] for i in order]
    elif ncol == 3 and len(handles) == 5:
        # 3 columns x 3 rows, filled column-major, with invisible padding
        # entries so the layout reads: entries 0/1 top+bottom of the left
        # column, entries 3/4 top+bottom of the middle column, and entry 2 at
        # the top of the right column.
        order = [0, None, 1, 3, None, 4, 2, None, None]
        handles = [blank if i is None else handles[i] for i in order]
        labels = ["" if i is None else labels[i] for i in order]

    # Draw on an oversized canvas first so the legend is never clipped, then
    # crop to the requested width (or wider, if the legend does not fit).
    fig = plt.figure(figsize=(max(width_in or 0.0, 24.0), 2.0))
    leg = fig.legend(handles, labels, loc="center", ncol=ncol,
                     fontsize=fontsize, frameon=True, edgecolor="grey",
                     framealpha=0.92, columnspacing=1.0, handlelength=1.6,
                     handletextpad=0.4, borderpad=0.4, borderaxespad=0.0)
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if width_in:
        canvas_w = max(width_in, bbox.width + 0.20)
        cx = 0.5 * (bbox.x0 + bbox.x1)
        bbox = Bbox([[cx - 0.5 * canvas_w, bbox.y0 - 0.05],
                     [cx + 0.5 * canvas_w, bbox.y1 + 0.05]])
    else:
        bbox = bbox.expanded(1.04, 1.10)
    fig_dir = R.HERE / "28GHz_LoS" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = fig_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches=bbox)
        print(f"Saved: {p}")
    plt.close(fig)


def replot(ddim_n, plot_taus, compact=False, figsize=None,
           separate_legend=False, legend_ncol=1, legend_width=None):
    n_real = ddim_n
    npb_list = R.NUM_PROBING_BEAM_PAIRS

    real_frob_snrs = [cached_snr(f"realonly_frob_N{n_real}", k) for k in npb_list]
    aug_snrs = {
        tau: [cached_snr(f"aug_real{n_real}_gen{R.GEN_SIZE}_tau{tau}", k)
              for k in npb_list]
        for tau in plot_taus
    }

    # Baselines depend only on the test split, which is deterministic in SEED.
    H, coords = R.load_or_cache_raw()
    ddim_coords = load_ddim_train_coords(str(R.LOGS_BASE), ddim_n)
    seen = set(coord_keys(ddim_coords)) if len(ddim_coords) else set()
    split = R.build_master_split(H, coords, seen, n_test=R.N_TEST, seed=R.SEED)
    baselines = compute_baselines(split["H_test"], R.SYS)
    bl_mrt, bl_genie = baselines["MRT_MRC"], baselines["genie_DFT"]

    npb = np.array(npb_list)
    # Compact mode: put the ticks at equally spaced positions instead of at
    # their true values, so the wide 8 -> 12 gap no longer stretches the axis.
    x = np.arange(len(npb_list), dtype=float) if compact else npb
    if figsize is None:
        # 4.952 in wide is chosen so that, after the tight crop, the panel is
        # saved 4.851 in wide: placed at 0.49\linewidth in an IEEE column it
        # gets the same 0.354 scale factor as a (10, 6) figure at \linewidth,
        # so 20 pt / 16 pt render at the same size as in the other figures.
        figsize = (4.952, 4.6) if compact else (10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(bl_mrt, color="black", ls="-", lw=2.0, zorder=4,
               label="MRT + MRC (upper bound)")
    ax.axhline(bl_genie, color="black", ls="--", lw=2.0, zorder=4,
               label="Genie-aided DFT beam pair")
    ax.plot(x, real_frob_snrs, color="#8c564b", ls=":", lw=2.0, marker="o",
            ms=6, mfc="none", zorder=3, label=f"Real Only (N={n_real})")
    for tau in plot_taus:
        ax.plot(x, aug_snrs[tau], zorder=3, mfc="none",
                label=rf"Real {n_real} + Gen {R.GEN_SIZE}, $\tau$={tau}",
                **R.TAU_STYLE[tau])

    ax.set_xlabel(r"$N_{\mathrm{probe}}$", fontsize=20)
    ax.set_ylabel("Average SNR (dB)", fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in npb], fontsize=20)
    if compact:
        ax.set_xlim(x[0] - 0.25, x[-1] + 0.25)
    ax.tick_params(axis="y", labelsize=20)
    ax.grid(True, alpha=0.30, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if not separate_legend:
        if ddim_n in (100, 1000):
            ax.legend(loc="lower right", fontsize=16, frameon=True,
                      edgecolor="grey", framealpha=0.92,
                      bbox_to_anchor=(1.0, 0.08))
        else:
            ax.legend(loc="center right", fontsize=16, frameon=True,
                      edgecolor="grey", framealpha=0.92)

    fig.tight_layout()
    fig_dir = R.HERE / "28GHz_LoS" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = f"beam_alignment_N{ddim_n}" + ("_compact" if compact else "")
    if separate_legend:
        stem += "_nolegend"
    for ext in ("png", "pdf"):
        p = fig_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)

    if separate_legend:
        save_legend_only(handles, labels,
                         f"beam_alignment_N{ddim_n}_legend", ncol=legend_ncol,
                         width_in=legend_width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddim_n", type=int, nargs="+", default=[100, 1000])
    ap.add_argument("--plot_taus", type=int, nargs="+", default=None)
    ap.add_argument("--compact", action="store_true",
                    help="equal-spaced x ticks + narrow canvas, saved as "
                         "beam_alignment_N<n>_compact.png|pdf")
    ap.add_argument("--figsize", type=float, nargs=2, default=None,
                    help="override the figure size in inches, e.g. 5.2 4.6")
    ap.add_argument("--separate_legend", action="store_true",
                    help="draw the panels without a legend and write the "
                         "legend to beam_alignment_N<n>_legend.png|pdf")
    ap.add_argument("--legend_ncol", type=int, default=2,
                    help="columns in the standalone legend (default: 2, which "
                         "gives 3 entries in the first column and 2 in the "
                         "second)")
    ap.add_argument("--legend_width", type=float, default=8.62,
                    help="minimum saved width (in) of the standalone legend; "
                         "the canvas is widened automatically if the legend "
                         "does not fit. 8.62 makes the N=100 and N=1000 "
                         "legends identical in width. Use 0 for a tight crop.")
    args = ap.parse_args()
    plot_taus = args.plot_taus if args.plot_taus else R.PLOT_TAUS
    figsize = tuple(args.figsize) if args.figsize else None
    legend_width = args.legend_width if args.legend_width else None
    for n in args.ddim_n:
        replot(n, plot_taus, compact=args.compact, figsize=figsize,
               separate_legend=args.separate_legend,
               legend_ncol=args.legend_ncol, legend_width=legend_width)


if __name__ == "__main__":
    main()
