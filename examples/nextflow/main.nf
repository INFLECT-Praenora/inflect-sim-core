#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// A vivarium composite run as one stage of a Nextflow pipeline. The point is
// interop: vivarium slots into existing workflow tooling, and owns the
// fine-grained, in-process coupling between engines that a task scheduler
// cannot express. Swap `spec` for any workflow spec — or use `viva-sim run
// <composite_id>` for a registered composite (e.g. the sepsis model).

params.spec  = "${projectDir}/chain.json"
params.steps = 1
params.viva  = "viva-sim"   // installed alias for: python -m viva_simulation_core.cli

process VIVA_RUN {
    input:
    path spec

    output:
    path "result.txt"

    script:
    """
    ${params.viva} workflow ${spec} --steps ${params.steps} --override inc.x=5.0 > result.txt
    """
}

process CONSUME {
    input:
    path result

    output:
    path "summary.txt"

    script:
    """
    grep 'scale.b' ${result} > summary.txt
    cat summary.txt
    """
}

workflow {
    VIVA_RUN(file(params.spec)) | CONSUME
}
