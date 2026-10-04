#!/usr/bin/env python3
"""Plot average beam-alignment SNR versus DDIM tau with the measured window."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from . import config as C
    from . import window_data as W
except ImportError:
    import config as C
    import window_data as W


OPTIMUM_FIELDS = [
    "N",
    "N_probe",
    "tau_star",
    "snr_star_db",
    "tau_gen",
    "tau_mem",
    "regime_of_optimum",
    "kappa",
    "fmem_threshold",
    "tie_count",
    "tied_taus",
    "plot_data_mode",
    "override_file",
]

EFFECTIVE_POINT_FIELDS = [
    "N",
    "N_probe",
    "tau",
    "average_snr_db",
    "point_source",
    "original_snr_db",
    "operation",
    "override_file",
]

FS_LABEL = 20
FS_TICK = 20
FS_LEGEND = 16


def _one(rows: list[dict[str, str]], experiment_type: str) -> dict[str, str]:
    matches = [row for row in rows if row["experiment_type"] == experiment_type]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {experiment_type} result, found {len(matches)}."
        )
    return matches[0]


def _load_overrides(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Manual plot-override table is missing: {path}")
    rows = W.read_csv_rows(path)
    required = {"N", "N_probe", "tau", "operation", "average_snr_db"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"Override table lacks required columns {required}: {path}")
    seen: set[tuple[int, int, int]] = set()
    for row in rows:
        key = (W.as_int(row["N"]), W.as_int(row["N_probe"]), W.as_int(row["tau"]))
        if key in seen:
            raise RuntimeError(f"Duplicate manual plot override for {key}")
        seen.add(key)
        operation = row["operation"].strip().lower()
        if operation not in {"set", "drop"}:
            raise ValueError(f"Unsupported override operation '{operation}' for {key}")
        if key[2] <= 0:
            raise ValueError(f"Manual tau must be positive for log plotting: {key}")
        if operation == "set" and not np.isfinite(W.as_float(row["average_snr_db"])):
            raise ValueError(f"Non-finite manual SNR for {key}")
    return rows


def _plot_configuration(
    all_rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
    n: int,
    n_probe: int,
    dry_run: bool,
    override_rows: list[dict[str, str]],
    override_path: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [
        row
        for row in all_rows
        if W.as_int(row["N"]) == n
        and W.as_int(row["N_probe"]) == n_probe
        and row.get("status") == "complete"
    ]
    augmented = sorted(
        [row for row in rows if row["experiment_type"] == "augmented"],
        key=lambda row: W.as_int(row["tau"]),
    )
    expected_taus = sorted(
        W.as_int(row["tau"]) for row in selected_rows if W.as_int(row["N"]) == n
    )
    actual_taus = [W.as_int(row["tau"]) for row in augmented]
    if actual_taus != expected_taus:
        missing = sorted(set(expected_taus) - set(actual_taus))
        extra = sorted(set(actual_taus) - set(expected_taus))
        raise RuntimeError(
            f"N={n}, N_probe={n_probe} augmented grid is incomplete. "
            f"Missing={missing}, extra={extra}"
        )

    real = _one(rows, "real_only")
    mrt = _one(rows, "MRT_MRC")
    genie = _one(rows, "genie_DFT")
    points: dict[int, dict[str, Any]] = {}
    for row in augmented:
        point_tau = W.as_int(row["tau"])
        point_snr = W.as_float(row["average_snr_db"])
        points[point_tau] = {
            "average_snr_db": point_snr,
            "point_source": "measured",
            "original_snr_db": point_snr,
            "operation": "none",
        }

    configuration_overrides = [
        row
        for row in override_rows
        if W.as_int(row["N"]) == n and W.as_int(row["N_probe"]) == n_probe
    ]
    for override in configuration_overrides:
        point_tau = W.as_int(override["tau"])
        operation = override["operation"].strip().lower()
        if operation == "drop":
            if point_tau not in points:
                raise RuntimeError(
                    f"Cannot drop absent point N={n}, N_probe={n_probe}, tau={point_tau}"
                )
            del points[point_tau]
            continue
        new_snr = W.as_float(override["average_snr_db"])
        if point_tau in points:
            original = points[point_tau]["original_snr_db"]
            source = "manual_replacement"
        else:
            original = ""
            source = "manual_addition"
        points[point_tau] = {
            "average_snr_db": new_snr,
            "point_source": source,
            "original_snr_db": original,
            "operation": "set",
        }
    if not points:
        raise RuntimeError(f"Manual overrides removed every point for N={n}, N_probe={n_probe}")

    effective_taus = sorted(points)
    tau = np.asarray(effective_taus, dtype=float)
    snr = np.asarray([points[value]["average_snr_db"] for value in effective_taus])
    if not np.isfinite(snr).all():
        raise ValueError(f"Non-finite SNR found for N={n}, N_probe={n_probe}")
    tau_gen = W.as_float(augmented[0]["tau_gen"])
    tau_mem = W.as_float(augmented[0]["tau_mem"])
    if any(
        not np.isclose(W.as_float(row["tau_gen"]), tau_gen)
        or not np.isclose(W.as_float(row["tau_mem"]), tau_mem)
        for row in augmented
    ):
        raise RuntimeError(f"Inconsistent window values for N={n}, N_probe={n_probe}")

    best_value = float(np.max(snr))
    tied = tau[np.isclose(snr, best_value, rtol=0.0, atol=1e-12)]
    tau_star = float(np.min(tied))
    best_index = int(np.where(tau == tau_star)[0][0])
    regime = (
        "before"
        if tau_star < tau_gen
        else "inside"
        if tau_star < tau_mem
        else "after"
    )
    summary = {
        "N": n,
        "N_probe": n_probe,
        "tau_star": int(tau_star),
        "snr_star_db": best_value,
        "tau_gen": tau_gen,
        "tau_mem": tau_mem,
        "regime_of_optimum": regime,
        "kappa": C.KAPPA,
        "fmem_threshold": C.FMEM_THRESHOLD,
        "tie_count": int(len(tied)),
        "tied_taus": ";".join(str(int(value)) for value in tied),
        "plot_data_mode": "manual_override" if configuration_overrides else "measured",
        "override_file": C.project_relative(override_path) if configuration_overrides else "",
    }
    effective_rows = [
        {
            "N": n,
            "N_probe": n_probe,
            "tau": point_tau,
            "average_snr_db": points[point_tau]["average_snr_db"],
            "point_source": points[point_tau]["point_source"],
            "original_snr_db": points[point_tau]["original_snr_db"],
            "operation": points[point_tau]["operation"],
            "override_file": (
                C.project_relative(override_path) if configuration_overrides else ""
            ),
        }
        for point_tau in effective_taus
    ]
    if dry_run:
        print(
            f"  N={n}, N_probe={n_probe}: {len(tau)} tau points, "
            f"best tau={int(tau_star)} ({best_value:.3f} dB), {regime} window, "
            f"manual overrides={len(configuration_overrides)}"
        )
        return summary, effective_rows

    C.ensure_output_directories()
    fig, ax = plt.subplots(figsize=(10, 6))
    window_label = "Generalization window"
    augmented_label = (
        rf"$N_{{\mathrm{{BA}}}}$ ({n}) + "
        rf"$N_{{\mathrm{{syn}}}}$ ({C.K_TOTAL - n})"
    )
    mrt_label = "MRT+MRC"
    genie_label = "Genie-aided DFT"
    real_only_label = rf"$N_{{\mathrm{{BA}}}}={n}$"
    ax.axvspan(
        tau_gen,
        tau_mem,
        color="#2ca02c",
        alpha=0.14,
        zorder=0,
        label=window_label,
    )
    ax.plot(
        tau,
        snr,
        "-o",
        color="#1f77b4",
        lw=2.2,
        ms=7,
        mfc="white",
        zorder=4,
        label=augmented_label,
    )
    mrt_snr = W.as_float(mrt["average_snr_db"])
    ax.axhline(
        mrt_snr,
        color="black",
        ls="-",
        lw=2.0,
        zorder=3,
        label=mrt_label,
    )
    ax.axhline(
        W.as_float(genie["average_snr_db"]),
        color="black",
        ls="--",
        lw=2.0,
        zorder=3,
        label=genie_label,
    )
    ax.axhline(
        W.as_float(real["average_snr_db"]),
        color="#8c564b",
        ls=":",
        lw=2.2,
        zorder=3,
        label=real_only_label,
    )
    ax.plot(
        [tau_star],
        [snr[best_index]],
        marker="*",
        ms=20,
        color="#d62728",
        zorder=6,
        label="_nolegend_",
    )

    ax.set_xscale("log")
    x_min = min(float(np.min(tau)), tau_gen) / 1.15
    x_max = max(float(np.max(tau)), tau_mem) * 1.15
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(r"$\tau$", fontsize=FS_LABEL)
    ax.set_ylabel("Average SNR (dB)", fontsize=FS_LABEL)
    if n == 500 and n_probe == 4:
        ax.set_yticks([18, 20, 22, 24])
    ax.tick_params(axis="both", which="both", labelsize=FS_TICK)
    ax.grid(True, which="both", alpha=0.30, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    handles_by_label = dict(zip(labels, handles))
    legend_order = [
        window_label,
        mrt_label,
        genie_label,
        real_only_label,
        augmented_label,
    ]
    ax.legend(
        [handles_by_label[label] for label in legend_order],
        legend_order,
        loc="upper left",
        bbox_to_anchor=(0.01, mrt_snr - 0.10),
        bbox_transform=ax.get_yaxis_transform(),
        borderaxespad=0.0,
        fontsize=FS_LEGEND,
        frameon=True,
        edgecolor="grey",
        framealpha=0.92,
    )
    fig.tight_layout()

    stem = f"beam_alignment_snr_vs_tau_N{n}_nprobe{n_probe}_kappa1_4_fmem20"
    for extension in ("png", "pdf"):
        output = C.FIGURE_DIR / f"{stem}.{extension}"
        C.assert_output_path(output)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"  Saved: {output}")
    plt.close(fig)
    print(
        f"  Optimum N={n}, N_probe={n_probe}: tau={int(tau_star)}, "
        f"SNR={best_value:.3f} dB, {regime.upper()} window"
    )
    return summary, effective_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(C.REFERENCE_SIZES))
    parser.add_argument("--n_probes", type=int, nargs="+", default=list(C.N_PROBES))
    parser.add_argument(
        "--overrides",
        type=Path,
        default=None,
        help="Manual set/drop table. Defaults to metrics/manual_plot_overrides.csv "
        "when that file exists.",
    )
    parser.add_argument(
        "--ignore_manual_overrides",
        action="store_true",
        help="Plot the measured run_results.csv values even if the default override file exists.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Validate without plotting")
    args = parser.parse_args()

    sizes = sorted(set(args.sizes))
    n_probes = sorted(set(args.n_probes))
    if set(sizes) - set(C.REFERENCE_SIZES):
        parser.error(f"Supported sizes are {list(C.REFERENCE_SIZES)}")
    if set(n_probes) - set(C.N_PROBES):
        parser.error(f"Supported N_probe values are {list(C.N_PROBES)}")
    if args.ignore_manual_overrides and args.overrides is not None:
        parser.error("Use either --overrides or --ignore_manual_overrides, not both.")
    if not C.RUN_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Run-level results are missing: {C.RUN_RESULTS_CSV}. Run run_experiment.py first."
        )
    all_rows = W.read_csv_rows(C.RUN_RESULTS_CSV)
    selected = W.load_selected_rows(sizes)
    override_path = None
    if not args.ignore_manual_overrides:
        override_path = args.overrides
        if override_path is None and C.PLOT_OVERRIDE_CSV.exists():
            override_path = C.PLOT_OVERRIDE_CSV
        elif override_path is not None and not override_path.is_absolute():
            override_path = Path.cwd() / override_path
    overrides = _load_overrides(override_path)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    print("=" * 78)
    print("28 GHz LoS beam-alignment SNR-vs-tau plots")
    print("=" * 78)
    print(
        f"Plot data     : {'manual overrides from ' + str(override_path) if override_path else 'measured results only'}"
    )
    summaries: list[dict[str, Any]] = []
    effective_points: list[dict[str, Any]] = []
    for n in sizes:
        for n_probe in n_probes:
            summary, points = _plot_configuration(
                all_rows,
                selected,
                n,
                n_probe,
                args.dry_run,
                overrides,
                override_path,
            )
            summaries.append(summary)
            effective_points.extend(points)

    if args.dry_run:
        print("\nPlot preflight passed; no files were written.")
        return

    old = W.read_csv_rows(C.OPTIMUM_CSV) if C.OPTIMUM_CSV.exists() else []
    active = {(n, n_probe) for n in sizes for n_probe in n_probes}
    preserved = [
        row
        for row in old
        if (W.as_int(row["N"]), W.as_int(row["N_probe"])) not in active
    ]
    final_rows: list[dict[str, Any]] = preserved + summaries
    final_rows.sort(key=lambda row: (W.as_int(row["N"]), W.as_int(row["N_probe"])))
    W.atomic_write_csv(C.OPTIMUM_CSV, OPTIMUM_FIELDS, final_rows)

    old_points = (
        W.read_csv_rows(C.EFFECTIVE_PLOT_POINTS_CSV)
        if C.EFFECTIVE_PLOT_POINTS_CSV.exists()
        else []
    )
    preserved_points = [
        row
        for row in old_points
        if (W.as_int(row["N"]), W.as_int(row["N_probe"])) not in active
    ]
    final_points: list[dict[str, Any]] = preserved_points + effective_points
    final_points.sort(
        key=lambda row: (
            W.as_int(row["N"]),
            W.as_int(row["N_probe"]),
            W.as_int(row["tau"]),
        )
    )
    W.atomic_write_csv(
        C.EFFECTIVE_PLOT_POINTS_CSV,
        EFFECTIVE_POINT_FIELDS,
        final_points,
    )
    print(f"\nOptimum summary: {C.OPTIMUM_CSV}")
    print(f"Effective points: {C.EFFECTIVE_PLOT_POINTS_CSV}")


if __name__ == "__main__":
    main()
