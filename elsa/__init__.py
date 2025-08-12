"""
ELSA: Embedding Locus Shingle Alignment

Order-aware syntenic-block discovery from protein language-model embeddings.
"""

__version__ = "0.1.0"
__author__ = "Claude & Jacob"

# Lazy imports to avoid loading heavy dependencies on import
def __getattr__(name):
    if name == "params":
        from . import params
        return params
    elif name == "manifest":
        from . import manifest
        return manifest
    elif name == "embeddings":
        from . import embeddings
        return embeddings
    elif name == "ingest":
        from . import ingest
        return ingest
    elif name == "projection":
        from . import projection
        return projection
    elif name == "shingling":
        from . import shingling
        return shingling
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")