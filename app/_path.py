"""Ensure the project root (parent of ``app/``) is on ``sys.path``.

Streamlit executes page scripts with their own directory on ``sys.path``,
so ``import configs`` / ``import src`` fail when running ``streamlit run
app/Home.py`` from the project root. Import this module as the first thing
in ``app/Home.py`` and every ``app/pages/*.py`` before any local imports.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
