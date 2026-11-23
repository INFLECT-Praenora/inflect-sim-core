#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// The REAL sepsis composite as a stage of a Nextflow pipeline (not the toy
// MathExpressionStep chain in main.nf). The point is interop at scale: Nextflow
// fans the sepsis core across a severity grid as independent tasks, and each task
// runs the vivarium composite through the same `viva-sim run` entrypoint the rest
// of the simulation core uses. Nextflow owns the outer data-parallel DAG; vivarium
// owns the in-process engine composition inside each task.
//
//   nextflow run sepsis.nf                       // default severity grid {0,1,3}
//   nextflow run sepsis.nf --severities 0,1,2,3  // your own grid
//   nextflow run sepsis.nf --steps 600           // longer horizon per task

params.severities = "0,1,3"
params.steps      = 360
params.python     = "python"   // a Python with the simulation-core package installed

// One task per severity: run the real sepsis_sbml composite under Tellurium and
// print its final clinical observables (MAP / TI / PB).
process SEPSIS_RUN {
    tag "severity=${sev}"

    input:
    val sev

    output:
    tuple val(sev), path("sev_${sev}.txt")

    script:
    """
    ${params.python} ${projectDir}/run_sepsis_task.py \
        --severity ${sev} --steps ${params.steps} > sev_${sev}.txt
    """
}

// Consume every task's output and score it into one clinical table — the
// data-parallel gather Nextflow is good at, over results vivarium produced.
process SUMMARIZE {
    publishDir ".", mode: "copy"

    input:
    path results

    output:
    path "sepsis_summary.tsv"

    script:
    """
    python ${projectDir}/summarize_sepsis.py ${results} > sepsis_summary.tsv
    cat sepsis_summary.tsv
    """
}

workflow {
    grid = Channel.fromList(params.severities.tokenize(",").collect { it.trim() as Integer })
    runs = SEPSIS_RUN(grid)
    SUMMARIZE(runs.map { sev, f -> f }.collect())
}
