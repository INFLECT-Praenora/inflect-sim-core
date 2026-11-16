# Provenance: BioGears sepsis core (Antimony / SBML)

## Files
- `biogears_sepsis.ant` - the model in Antimony (706 lines, 91 state variables).
- `biogears_sepsis.xml` - the same model exported to SBML (L3V2).
- `BioGears sepsis core in Antimony extraction report.pdf` / `.docx` - the extraction report (same content, two formats).

## Model
**BioGears sepsis core**, extracted into Antimony (706 lines, 91 state variables), which also exports to SBML (L3V2).

## Author
Extraction and report by **Herbert Sauro** (report dated Nov 13 2026).

## Source
Translated from the **BioGears physiology engine** source (Apache License 2.0, github.com/BioGearsEngine/core): cardiovascular circuit, tissue fluid exchange with albumin transport, baroreflex, and the Acute Inflammatory Response model.

## Scientific reference
McDaniel et al. 2019, "A Whole-Body Mathematical Model of Sepsis Progression and Treatment Designed in the BioGears Physiology Engine," Front. Physiol. 10:1321, doi:10.3389/fphys.2019.01321.

## Patient
BioGears StandardMale (77.1 kg, HR 72, BP 114/73.5).

## Parameters
`severity` 1/2/3 (mild/moderate/severe), `infectionStart` (s, >0 to start), `antibiotic`, `Q_IV` (IV fluid mL/s), `IValb` (albumin g/mL), `infScale`.

## Known quirks and caveats (from the report)
- Keeps BioGears' `(1-PT)` tissue-pathogen term (where `(1-PT/uP)` was likely intended).
- SI/LI/splanchnic merged into one gut node.
- Severe infection seeds at 2.5e7 bacteria (BioGears code value; the paper states 1e7).
- No urine, respiration, gas exchange, lactate, or intracellular water; single well-mixed albumin pool.
- Validation in the report was against the McDaniel 2019 paper milestones only; BioGears was not run by the author.

## License
The `.ant` and `.xml` are derivatives of Apache-2.0 BioGears. The report is Sauro's authored work, redistributed here with permission.
