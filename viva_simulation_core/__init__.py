"""viva_simulation_core — workspace Python package.

Importing the package pulls in ``.composites`` so the ``@composite_generator``
decorators fire and the workspace's composites register in process-bigraph's
generator registry (the dashboard manifest reads that registry). Without this,
an editable install's empty ``top_level.txt`` hides the package from the
dist-scan and the composites never resolve.
"""

from viva_simulation_core import composites  # noqa: F401 — register generators on import

__all__ = ["composites"]
