"""
Concrete labeler classes.
"""

from .ideal_labeler import IdealLabeler
try:
    from .interactive_labeler import InteractiveLabeler
except ImportError:
    raise ImportError("Error importing matplotlib."
                      "InteractiveLabeler not supported.")
