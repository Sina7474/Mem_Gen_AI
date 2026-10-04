"""Generalisation windows in diffusion-based wireless channel synthesis.

``memgen`` trains unconditional DDIMs on site-specific MIMO channels and
measures, as a function of training time, both how well the generated channels
match the true distribution and how often they reproduce individual training
channels. The band in which the first has converged and the second has not yet
begun is the *generalisation window*.

Typical use from Python::

    from memgen import train, evaluate, figures

    run_dir = train.train(dataset="sionna_3p5ghz", n=1000)
    table = evaluate.evaluate_sweep("dataset-size")
    figures.fidelity_and_memorisation(table)

The same operations are available from the command line via ``memgen --help``.
"""

__version__ = "1.0.0"

__all__ = [
    "beamspace",
    "config",
    "datasets",
    "downstream",
    "evaluate",
    "figures",
    "metrics",
    "model",
    "phase",
    "sampling",
    "train",
]
