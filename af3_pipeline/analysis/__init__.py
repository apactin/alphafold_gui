"""
af3_pipeline.analysis package
=============================
Post-AF3 analysis modules:
- post_analysis: orchestrates the full pipeline
- build_meta: extracts metadata (bond info, seeds, etc.)
- prepare_structures: generates kept/broken variants
- rosetta_minimize: runs Rosetta relax/minimization
- metrics: computes scores, RMSD, ΔE
"""

__all__ = [
    "post_analysis",
    "build_meta",
    "prepare_structures",
    "rosetta_minimize",
    "metrics",
]

from importlib import import_module

def __getattr__(name):
    if name in __all__:
        return import_module(f"af3_pipeline.analysis.{name}")
    raise AttributeError(f"module 'af3_pipeline.analysis' has no attribute '{name}'")
