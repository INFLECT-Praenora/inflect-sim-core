"""simulation-core composites.

Importing this subpackage fires the ``@composite_generator`` decorators so the
workspace's own composites register in process-bigraph's generator registry and
the dashboard manifest can resolve them (same pattern as viva-biogears'
``composites/__init__``).
"""

from viva_simulation_core.composites.sepsis_sbml import (
    SEPSIS_OBSERVABLES,
    make_sepsis_sbml_composite,
)
from viva_simulation_core.composites.sepsis_cosim import (
    make_sepsis_cosim_composite,
    extract_cosim,
)

# Register every curated model from inflect-models-db as a composite generator, so
# the catalog/workbench lists them alongside the workspace's own composites. This is
# a no-op when inflect-models-db isn't installed (optional, lazy — engine-free).
from viva_simulation_core import models_db as _models_db  # noqa: E402
_models_db.register_all()

__all__ = [
    "make_sepsis_sbml_composite",
    "SEPSIS_OBSERVABLES",
    "make_sepsis_cosim_composite",
    "extract_cosim",
]
