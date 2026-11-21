"""SurrogateProcess — wrap a fitted surrogate as a process-bigraph Process so it is
an interchangeable engine box over the same typed observable ports (one flat port
per observable, the NeuralProcess convention). Swapping a real engine for a scored
surrogate becomes a one-node change in a composite.
"""
from __future__ import annotations

import numpy as np
from process_bigraph import Process


class SurrogateProcess(Process):
    config_schema = {"observables": "list[string]", "fitted": "any"}

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self.fitted = self.config["fitted"]                 # a FittedSurrogate
        self.observables = list(self.config["observables"])

    def inputs(self):
        return {name: "float" for name in self.observables}

    def outputs(self):
        return {name: "float" for name in self.observables}

    def update(self, state, interval):
        x = np.array([float(state.get(n, 0.0)) for n in self.observables], float)
        nxt = np.asarray(self.fitted.rollout(x, 2), float)[-1]
        return {n: float(nxt[i]) for i, n in enumerate(self.observables)}


def register(core, name: str = "SurrogateProcess") -> None:
    """Register SurrogateProcess into a core so it is addressable like any module."""
    core.register_link(name, SurrogateProcess)
