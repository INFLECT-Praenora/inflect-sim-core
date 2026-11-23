# Vivarium inside a Nextflow pipeline

Nextflow DSL2 pipelines whose stages run a **vivarium composite** and consume its
output. They show that vivarium drops into existing workflow tooling with no
special glue — the composite is just a command the scheduler runs.

## The real sepsis model (`sepsis.nf`)

The headline example: Nextflow fans the **real `sepsis_sbml` composite** across a
severity grid as independent tasks, then gathers the results into one clinical
table — the outer data-parallel DAG is Nextflow's, the in-process engine run
inside each task is vivarium's.

```
nextflow run sepsis.nf                         # severity grid {0,1,3}
nextflow run sepsis.nf --severities 0,1,2,3    # your own grid
nextflow run sepsis.nf --steps 600             # longer horizon per task
```

`SEPSIS_RUN` invokes `viva-sim run sepsis_sbml --param severity=<N> --emit
parameters/MAP,parameters/TI,parameters/PB` (one task per severity); `SUMMARIZE`
gathers every task's final state and scores each for septic shock
(`sepsis_summary.tsv`). Needs the `tellurium` extra installed.

## The minimal example (`main.nf`)

A two-stage toy chain (`MathExpressionStep`) for the mechanics only:

```
nextflow run main.nf
# or point it at your own spec / step count:
nextflow run main.nf --spec my_workflow.json --steps 20
```

`VIVA_RUN` invokes `viva-sim workflow <spec> --steps N` (the installed alias for
`python -m viva_simulation_core.cli`). Swap in `viva-sim run <composite_id>` to
run any registered composite.

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
