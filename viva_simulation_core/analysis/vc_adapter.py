"""Adapters so viva-core (the Rust execution backend) can instantiate and drive
TelluriumProcess. viva-core is NOT on the engine-core UniformTimeCourseStep
contract (unlike TelluriumFullReportUTCStep / CopasiFullReportUTCStep), so running
a model under the Rust engine still goes through a Tellurium-process adapter that
bridges two viva-core (Milestone-1) constraints, with no change to the underlying
roadrunner computation:

* viva-core's PyProcess calls ``Class(config)`` with no ``core`` argument, but
  ``bigraph_schema.Edge.__init__`` requires one, so a core is supplied.
* viva-core rejects an *empty* container in a process update ("op/store type
  mismatch"), and applies ``overwrite`` ops only to leaf stores, not to a whole
  map-valued store. So each emitted observable port is declared as a struct of
  per-id ``overwrite[float]`` leaves, and empty ports are dropped.

Two adapters:

* :class:`VivaCoreTelluriumProcess` — the original, specialized to the sepsis
  model, whose observables all live in ``parameters`` (no floating species).
* :class:`VivaCoreTelluriumGeneric` — works for *any* SBML model: it emits
  whichever of ``species`` / ``parameters`` is non-empty as per-id leaves, so a
  model whose observables are floating species (the common case) runs under
  viva-core too.
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
        # Guard the "always empty" claim: never silently drop real data.
        assert len(up['species']) == 0 and len(up['rates']) == 0, (
            'SBML model now has floating species/rates; viva-core adapter would drop them')
        up = {k: v for k, v in up.items() if k not in _DROP}
        up['parameters'] = {sid: float(up['parameters'][sid])
                            for sid in SEPSIS_OBSERVABLES.values()}
        return up


class VivaCoreTelluriumGeneric(TelluriumProcess):
    """Model-agnostic viva-core adapter for any SBML model with floating species.

    Unlike :class:`VivaCoreTelluriumProcess` (sepsis-only, which emits
    ``parameters`` because that model has no floating species), this emits the
    model's floating ``species`` — the observables for a typical SBML model — as
    per-id ``overwrite[float]`` leaves, keyed by SBML id, and drops ``rates`` /
    ``parameters`` so viva-core never sees an empty or whole-map container.
    """

    def __init__(self, config=None, core=None):
        super().__init__(config, core=core if core is not None else allocate_core())

    def outputs(self):
        out = {k: v for k, v in super().outputs().items()
               if k not in ('species', 'rates', 'parameters')}
        out['species'] = {sid: 'overwrite[float]' for sid in self.get_species_ids()}
        return out

    def update(self, state, interval=1.0):
        up = super().update(state, interval)
        ids = list(self.get_species_ids())
        assert ids, ('model has no floating species; use VivaCoreTelluriumProcess '
                     'for parameters-only models (e.g. the sepsis SBML)')
        arr = up['species']
        out = {k: v for k, v in up.items() if k not in ('species', 'rates', 'parameters')}
        out['species'] = {sid: float(arr[i]) for i, sid in enumerate(ids)}
        return out
