# Provenance: BioGears sepsis core (Antimony / SBML)

These files were provided for this investigation (source: project-supplied).

## Files
- `biogears_sepsis.ant` - Antimony form of the model (706 lines).
- `biogears_sepsis.xml` - SBML export of the same model (L3V2, per the report).
- `BioGears sepsis core in Antimony extraction report.pdf` / `.docx` - the extraction report (same content, two formats).

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
