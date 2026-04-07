"""
ELSA: Embedding Locus Search and Alignment

Syntenic-block discovery from protein language-model embeddings via
gene-level anchor chaining.
"""

__version__ = "2.0.0"
__author__ = "Jacob Westbrook, with assistance from Claude (Anthropic) and GPT (OpenAI)"

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
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
