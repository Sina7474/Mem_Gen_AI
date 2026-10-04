"""Command-line interface.

Every experiment in the paper is reachable through one of six subcommands::

    memgen train         train a DDIM and checkpoint it along the tau grid
    memgen evaluate      fidelity and memorisation versus tau for an ablation
    memgen phase         fit the generalisation-memorisation phase boundary
    memgen figure        redraw a paper figure from a CSV table
    memgen csi           CSI-compression downstream experiment
    memgen beam          beam-alignment downstream experiment

Run ``memgen <command> --help`` for the arguments of each.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config as cfg

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memgen",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Paths are configurable through MEMGEN_DATA_ROOT, "
               "MEMGEN_RUN_ROOT and MEMGEN_RESULT_ROOT.",
    )
    parser.add_argument("--version", action="store_true",
                        help="print the package version and exit")
    sub = parser.add_subparsers(dest="command", metavar="command")

    datasets = sorted(cfg.DATASETS)

    # ------------------------------------------------------------------ train
    train = sub.add_parser("train", help="train a DDIM")
    train.add_argument("-d", "--dataset", choices=datasets,
                       default=cfg.DEFAULT_DATASET)
    train.add_argument("-n", "--size", type=int, required=True,
                       help="number of training channels N")
    train.add_argument("-w", "--width", type=int, default=cfg.DEFAULT_WIDTH,
                       help="U-Net base width W")
    train.add_argument("-b", "--batch-size", type=int, default=None,
                       help="mini-batch size (default min(N, 500))")
    train.add_argument("--full-batch", action="store_true",
                       help="use B = N, the noise-free gradient control")
    train.add_argument("--max-tau", type=int, default=cfg.DEFAULT_MAX_TAU,
                       help="number of optimiser steps")
    train.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    train.add_argument("--no-resume", action="store_true",
                       help="start from scratch even if checkpoints exist")

    # --------------------------------------------------------------- evaluate
    evaluate = sub.add_parser("evaluate",
                              help="FCD and f_mem versus tau for an ablation")
    evaluate.add_argument("sweep", nargs="?", default="dataset-size",
                          choices=sorted(_sweeps()),
                          help="which ablation to evaluate")
    evaluate.add_argument("-d", "--dataset", choices=datasets,
                          default=cfg.DEFAULT_DATASET)
    evaluate.add_argument("--sizes", type=int, nargs="+")
    evaluate.add_argument("--widths", type=int, nargs="+")
    evaluate.add_argument("--batch-sizes", type=int, nargs="+")
    evaluate.add_argument("--num-generated", type=int, default=cfg.N_GENERATED,
                          help="channels sampled per checkpoint")
    evaluate.add_argument("--kappa", type=float, default=cfg.KAPPA,
                          help="nearest-neighbour ratio threshold for f_mem")
    evaluate.add_argument("--weights", choices=["ema", "raw"], default="ema")
    evaluate.add_argument("-o", "--output", type=Path)

    # ------------------------------------------------------------------ phase
    phase = sub.add_parser("phase", help="fit the phase boundary N_c(W)")
    phase.add_argument("table", type=Path,
                       help="CSV produced by `memgen evaluate model-size`")
    phase.add_argument("--eps", type=float, default=0.10,
                       help="memorisation level defining the boundary")
    phase.add_argument("--multipliers", type=float, nargs="+",
                       default=[1.0, 2.0, 5.0],
                       help="training times in units of tau_gen(W)")
    phase.add_argument("--reference-size", type=int, default=None,
                       help="N used to read off tau_gen(W)")
    phase.add_argument("--bootstrap", type=int, default=1_000)
    phase.add_argument("-o", "--output", type=Path)

    # ----------------------------------------------------------------- figure
    figure = sub.add_parser("figure", help="redraw a figure from a CSV table")
    figure.add_argument("kind", choices=["loss", "fidelity", "collapse",
                                         "phase", "downstream"])
    figure.add_argument("tables", type=Path, nargs="+")
    figure.add_argument("--by", default="N",
                        help="column separating the curves (N, width, ...)")
    figure.add_argument("--value", default="test_nmse_db",
                        help="metric column for the downstream figure")
    figure.add_argument("--ylabel", default="test NMSE [dB]")
    figure.add_argument("--name", default=None, help="output file stem")

    # -------------------------------------------------------------- downstream
    csi = sub.add_parser("csi", help="CSI-compression downstream experiment")
    csi.add_argument("-d", "--dataset", choices=datasets,
                     default=cfg.DEFAULT_DATASET)
    csi.add_argument("--sizes", type=int, nargs="+")
    csi.add_argument("--taus", type=int, nargs="+")
    csi.add_argument("--width", type=int, default=cfg.DEFAULT_WIDTH,
                     help="U-Net width of the generator runs to sample from")
    csi.add_argument("--budget", type=int, default=5_000,
                     help="total training channels K seen by every CRNet")
    csi.add_argument("--seeds", type=int, nargs="+")
    csi.add_argument("--epochs", type=int, default=500)
    csi.add_argument("-o", "--output", type=Path)

    beam = sub.add_parser("beam", help="beam-alignment downstream experiment")
    beam.add_argument("-d", "--dataset", choices=datasets, default="sionna_28ghz")
    beam.add_argument("--sizes", type=int, nargs="+")
    beam.add_argument("--taus", type=int, nargs="+")
    beam.add_argument("--probes", type=int, nargs="+",
                      help="numbers of probing beam pairs")
    beam.add_argument("--width", type=int, default=cfg.DEFAULT_WIDTH,
                      help="U-Net width of the generator runs to sample from")
    beam.add_argument("--budget", type=int, default=5_000)
    beam.add_argument("--seeds", type=int, nargs="+")
    beam.add_argument("--epochs", type=int, default=1_000)
    beam.add_argument("-o", "--output", type=Path)

    return parser


def _sweeps():
    from .evaluate import SWEEPS
    return SWEEPS


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__
        print(__version__)
        return 0
    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "train":
        from .train import train
        train(dataset=args.dataset, n=args.size, width=args.width,
              batch_size=args.batch_size, full_batch=args.full_batch,
              max_tau=args.max_tau, seed=args.seed, resume=not args.no_resume)

    elif args.command == "evaluate":
        from .evaluate import evaluate_sweep
        evaluate_sweep(sweep=args.sweep, dataset=args.dataset, sizes=args.sizes,
                       widths=args.widths, batch_sizes=args.batch_sizes,
                       n_generated=args.num_generated, kappa=args.kappa,
                       weights=args.weights, output=args.output)

    elif args.command == "phase":
        from .phase import build_boundary
        build_boundary(args.table, eps=args.eps,
                       multipliers=tuple(args.multipliers),
                       reference_n=args.reference_size,
                       n_boot=args.bootstrap, output=args.output)

    elif args.command == "figure":
        from . import figures
        tables = args.tables
        if args.kind == "loss":
            figures.loss_curves(tables, name=args.name or "loss_vs_tau")
        elif args.kind == "fidelity":
            figures.fidelity_and_memorisation(tables[0], by=args.by,
                                              name=args.name)
        elif args.kind == "collapse":
            figures.memorisation_collapse(tables[0], name=args.name)
        elif args.kind == "phase":
            figures.phase_diagram(tables[0], name=args.name or "phase_diagram")
        else:
            figures.downstream_curve(tables[0], value=args.value,
                                     ylabel=args.ylabel, name=args.name)

    elif args.command == "csi":
        from .downstream.csi_compression import run
        run(dataset=args.dataset, sizes=args.sizes, taus=args.taus,
            k_total=args.budget, seeds=args.seeds, epochs=args.epochs,
            width=args.width, output=args.output)

    elif args.command == "beam":
        from .downstream.beam_alignment import run
        run(dataset=args.dataset, sizes=args.sizes, taus=args.taus,
            n_probes=args.probes, k_total=args.budget, seeds=args.seeds,
            epochs=args.epochs, width=args.width, output=args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
