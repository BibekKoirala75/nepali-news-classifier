"""
Re-exports the exact preprocessing function used at training time
(ml/src/preprocessing.py) so the API applies identical cleaning logic
before handing text to the model. Do not reimplement cleaning here --
import it.
"""

import os
import sys

_ML_SRC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ml", "src"
)
_ML_SRC_DIR = os.path.normpath(_ML_SRC_DIR)
if _ML_SRC_DIR not in sys.path:
    sys.path.insert(0, _ML_SRC_DIR)

from preprocessing import clean_text, is_valid_text  # noqa: E402,F401
