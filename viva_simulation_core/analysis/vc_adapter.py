"""Adapter so viva-core can instantiate and drive TelluriumProcess.

Two viva-core (Milestone-1) constraints are bridged here, with no change to
the underlying roadrunner computation:

* viva-core's PyProcess calls ``Class(config)`` with no ``core`` argument, but
  ``bigraph_schema.Edge.__init__`` requires one, so a core is supplied.
* viva-core rejects an *empty* container in a process update ("op/store type
  mismatch"). The sepsis SBML model has no floating species and no rate
  entries, so ``species`` / ``rates`` are always empty; this adapter drops
  those two ports and emits only ``parameters`` and ``time``.
* viva-core applies ``overwrite`` ops only to leaf stores, not to a whole
  map-valued store (``op/store type mismatch``). The ``parameters`` port is
  therefore declared as a struct of per-id ``overwrite[float]`` leaves,
  restricted to the shared sepsis observables (the only values compared).
"""
from viva_simulation_core.composites.sepsis_sbml import SEPSIS_OBSERVABLES
from process_bigraph import allocate_core
from viva_tellurium.processes import TelluriumProcess

_DROP = ('species', 'rates')


class VivaCoreTelluriumProcess(TelluriumProcess):
    def __init__(self, config=None, core=None):
        super().__init__(config, core=core if core is not None else allocate_core())

    def outputs(self):
        out = {k: v for k, v in super().outputs().items() if k not in _DROP}
        out['parameters'] = {sid: 'overwrite[float]' for sid in SEPSIS_OBSERVABLES.values()}
        return out

    def update(self, state, interval=1.0):
        up = super().update(state, interval)
        up = {k: v for k, v in up.items() if k not in _DROP}
        up['parameters'] = {sid: float(up['parameters'][sid])
                            for sid in SEPSIS_OBSERVABLES.values()}
        return up
