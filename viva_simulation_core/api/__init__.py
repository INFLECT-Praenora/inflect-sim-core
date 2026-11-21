"""The simulation-core workflow API — pure functions over process-bigraph.

The four verbs live here (``list_modules``, ``module_card``, ``call_module``,
``spec_to_composite_state`` + ``WorkflowRegistry``) and the model helpers in
``modeling``. No server: the top-level facade (``viva_simulation_core``) and the
``viva-sim`` CLI are thin wrappers over these functions.
"""
