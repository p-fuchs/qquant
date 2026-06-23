"""Orchestration subpackage: vast.ai lifecycle wrapper + the go/no-go spike driver.

Import-time torch-free so it runs on the macOS dev laptop. GPU code lives only in
``scripts/spike/spike_remote.py`` (copied to the box and run as a script, not imported).

Spec 02 owns this package marker plus ``vastai`` and ``spike``. Later specs (e.g. Spec
08) *add* modules here and must not recreate this file.
"""

from __future__ import annotations
