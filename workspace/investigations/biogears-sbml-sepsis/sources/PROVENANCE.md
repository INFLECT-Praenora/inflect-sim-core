# Provenance: BioGears sepsis core (Antimony / SBML)

These files were provided for this investigation (source: project-supplied).

## Files
- `biogears_sepsis.ant` - Antimony form of the model (706 lines).
- `biogears_sepsis.xml` - SBML export of the same model (L3V2, per the report).
- `BioGears sepsis core in Antimony extraction report.pdf` / `.docx` - the extraction report (same content, two formats).
- `proto_valve.ant` / `proto_valve.xml` - **event-free prototype** of the same model (see below).

## Event-free prototype (`proto_valve`)
`proto_valve.{ant,xml}` is a derivative of `biogears_sepsis` (same `biogears_sepsis`
model id) with the two heartbeat-event sources of CVODE root returns removed, as a
speed experiment. Changes (per the Nov 23 2026 prototype report):

- `E_beat` (the per-beat event) becomes a continuous phase oscillator setting the
  heartbeat driver `eshape`.
- The piecewise pulmonic-valve gate `pulOpen` becomes per-artery valve gating.
- `SBP := MAP + 19.1` and `DBP := MAP - 22` (assignment rules) and 5 s filters for
  MAP/CVP/COmeas replace the per-beat latched pressures.
- Heart-rate changes act immediately instead of at the next beat.
- The three infection/TLR events (`E_tlrOn`, `E_tlrOff`, `E_infect`) are unchanged,
  so the `severity` / `infectionStart` scenario knobs are identical.

Measured behaviour (roadrunner 2.12.0 wheel): ~1.24–1.44x faster than the full model,
with per-beat hemodynamics within ~0.2 mmHg and immune variables within ~1e-6 (`Alb_v`
within ~1e-4). It is a **speed experiment, not a validated replacement** for `E_beat`:
the per-artery valve, the `SBP`/`DBP` offsets, and the 5 s filters are physiological
stand-ins, and instantaneous values do not line up (compare per-beat summaries). Run
in the simulation core via the `sepsis_sbml_eventfree_valve` composite. Same authorship,
license (Apache-2.0 derivative of BioGears), and caveats as the full model below.

## Model
**BioGears sepsis core**, extracted from the BioGears physiology engine into Antimony (91 state variables, per the report). Targets BioGears **8.2.0**, patient StandardMale (77.1 kg, HR 72, BP 114/73.5).

## Author
Extraction and report by **Herbert Sauro** (report dated Nov 13 2026).

## Source
Translated from the BioGears physiology engine source (github.com/BioGearsEngine/core): cardiovascular circuit, tissue fluid exchange with albumin transport, baroreflex, and the Acute Inflammatory Response model.

## Scientific reference
McDaniel et al. 2019, "A Whole-Body Mathematical Model of Sepsis Progression and Treatment Designed in the BioGears Physiology Engine," Front. Physiol. 10:1321, doi:10.3389/fphys.2019.01321.

## Parameters
`severity` 1/2/3 (mild/moderate/severe), `infectionStart` (s, >0 to start), `antibiotic`, `Q_IV` (IV fluid mL/s), `IValb` (albumin g/mL), `infScale`.

## Known quirks and caveats (per the report)
- Keeps BioGears' `(1-PT)` tissue-pathogen term (where `(1-PT/uP)` was likely intended).
- SI/LI/splanchnic merged into one gut node.
- Severe infection seeds at 2.5e7 bacteria (BioGears code value; the paper states 1e7).
- No urine, respiration, gas exchange, lactate, or intracellular water; single well-mixed albumin pool.
- Validation in the report was against the McDaniel 2019 paper milestones only; BioGears was not run by the author.

## License and notices

### Model files (`biogears_sepsis.ant`, `biogears_sepsis.xml`)
These two files are derivatives of BioGears and are **distributed under the Apache License 2.0**. See this repository's `LICENSE` file (Apache-2.0).

- Upstream copyright holder: **Applied Research Associates, Inc.** (BioGears, BioGearsEngine/core, Apache-2.0).
- Statement of changes (Apache-2.0 section 4): these files are a **modified work**. The BioGears sepsis core was translated from C++ into Antimony/SBML by Herbert Sauro; they are not a verbatim copy of BioGears source.

### Extraction report (`.pdf` / `.docx`)
The report was authored by **Herbert Sauro** and is included in this INFLECT `simulation-core` investigation at the direction of the repository owner / project lead. Author credit: Herbert Sauro. No license is asserted here for the report itself.
