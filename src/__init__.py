"""src package initialization.

This package provides core components for the Nomad Muse music generation system.
All CLI scripts are import-safe - they only execute when run directly.
"""

# Import only core modules that should be available as libraries
from . import vocab
from . import model
from . import data_prep

__all__ = [
    "vocab",
    "model",
    "data_prep",
]
