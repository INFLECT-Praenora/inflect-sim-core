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

`SEPSIS_RUN` runs one task per severity: `run_sepsis_task.py` builds and runs the
real `sepsis_sbml` composite and prints its final MAP, tissue integrity, and blood
pathogen. `SUMMARIZE` gathers every task's output and scores each for septic shock
into one table (`sepsis_summary.tsv`). Needs the `tellurium` extra installed, and a
`python` on PATH with the `simulation-core` package.

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

## The two layers are complementary

Nextflow and vivarium work at different levels, so you use both:

- **Nextflow** schedules coarse, independent tasks (each a container or script)
  across a DAG. It is good at data-parallel batch and provenance.
- **Vivarium** composes processes that share state and step together each timestep.
  This is the tight per-tick coupling a task scheduler cannot express — for example
  the physiology engine and the antibiotic controller exchanging state every tick in
  the co-simulation study (06).

Nextflow owns the outer DAG; vivarium owns the composite each task runs.
