"""Make the ``scripts/`` entry points importable for the test suite.

The scripts are standalone CLI modules (they import each other's functions
rather than a shared ``src/`` package), so tests import them off this path.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(_ROOT, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
