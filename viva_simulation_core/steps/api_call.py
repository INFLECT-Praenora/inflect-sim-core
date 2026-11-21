"""ApiCallStep — a process-bigraph Process whose update is a remote model call.

The "API-calling step": instead of computing locally, it POSTs the current values
of its declared observables to a model-serving ``url`` (e.g. ``serving.model_server``
or a remote viva-api) and returns the stepped state. This is how a vivarium
composite orchestrates a *remote* simulation as one interchangeable node.
"""
from __future__ import annotations

from process_bigraph import Process


class ApiCallStep(Process):
    config_schema = {
        "url": "string",
        "observables": "list[string]",
        "timeout": {"_type": "float", "_default": 5.0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self.url = self.config["url"]
        self.observables = list(self.config["observables"])
        self.timeout = float(self.config.get("timeout", 5.0))

    def inputs(self):
        return {name: "float" for name in self.observables}

    def outputs(self):
        return {name: "float" for name in self.observables}

    def update(self, state, interval):
        import requests

        overrides = {name: float(state.get(name, 0.0)) for name in self.observables}
        try:
            resp = requests.post(self.url, json={"overrides": overrides}, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"ApiCallStep: POST {self.url} failed: {e}") from e
        body = resp.json()
        # The server speaks absolute state (overrides in, stepped state out, like
        # serving.model_server). Composite float ports apply updates additively, so
        # return the delta that lands the store on the server's absolute value.
        return {name: float(body[name]) - overrides[name]
                for name in self.observables if name in body}
