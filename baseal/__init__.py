"""
BaseAL: a framework for benchmarking active learning sampling strategies on
pre-computed embeddings.

The main entry points are `ActiveLearner` (one sample-query-train loop) and
`Manager` (several `ActiveLearner` experiments defined in a YAML config).
"""
__version__ = "1.3.0"

__all__ = ["EmbeddingClassifier", "ActiveLearner", "Manager", "__version__"]

# Imported on first use rather than here, so that e.g.
# `from baseal.utils.sampling import SamplingStrategy` doesn't load PyTorch or
# trigger the UMAP warm-up in baseal.active_learner.
_LAZY = {
    "EmbeddingClassifier": "baseal.utils.model",
    "ActiveLearner": "baseal.active_learner",
    "Manager": "baseal.active_learner",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'baseal' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
