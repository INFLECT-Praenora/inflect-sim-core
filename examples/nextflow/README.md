# Vivarium inside a Nextflow pipeline

A Nextflow DSL2 pipeline whose first stage runs a **vivarium composite** and
whose second stage consumes its output. It shows that vivarium drops into
existing workflow tooling with no special glue — the composite is just a
command the scheduler runs.

```
nextflow run main.nf
# or point it at your own spec / step count:
nextflow run main.nf --spec my_workflow.json --steps 20
```

`VIVA_RUN` invokes `viva-sim workflow <spec> --steps N` (the installed alias for
`python -m viva_simulation_core.cli`). Swap in `viva-sim run <composite_id>` to
run a registered composite such as the sepsis model.

## Where the layers sit — and why this is interop, not a contest

Nextflow and vivarium operate at **different layers**, and that is the point:

- **Nextflow** schedules coarse, independent *tasks* (each a container or
  script) across a DAG — great for data-parallel batch and provenance.
- **Vivarium** composes *processes* that share state and step together at each
  timestep — the fine-grained, tightly-coupled engine composition (an SBML
  engine handing state to a surrogate handing state to a physiology engine on
  every tick) that a task scheduler's process model cannot express.

So this is **not** a "faster than Nextflow" claim — the two solve different
problems. Vivarium runs *inside* a Nextflow task when you want pipeline-level
orchestration, and owns the in-process coupling within each task. Use both:
Nextflow for the outer DAG, vivarium for the composite each stage runs.
