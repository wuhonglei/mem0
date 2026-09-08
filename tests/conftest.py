"""Ensure server/ is importable the same way Docker runs it (bare `from auth import`)."""

import os
import sys

_SERVER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "server")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)
