#!/usr/bin/env python3
"""Run STAGE on macOS by redirecting its Linux-only /dev/shm temp root."""

from __future__ import annotations

import sys

import symbolic_tensor_graph.graph.graph as graph_module


graph_module.TMP_DIR_ROOT = "/private/tmp"

from main import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
