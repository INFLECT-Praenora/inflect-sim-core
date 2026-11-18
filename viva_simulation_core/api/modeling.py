"""Model ingest, validation and observable inference (server-independent)."""

from __future__ import annotations

import hashlib
import re

__all__ = [
    "UnsafeModelError",
    "UnknownObservableError",
    "model_hash",
    "reject_unsafe",
    "load_engine",
    "infer_observables",
    "validate_observables",
    "tunable_parameters",
]

_FORMATS = ("antimony", "sbml")


class UnsafeModelError(ValueError):
    """Model source contains an import / external reference."""


class UnknownObservableError(ValueError):
    """Requested observable id is not present in the model."""


def _check_format(model_format: str) -> None:
    if model_format not in _FORMATS:
        raise ValueError(f"unknown model_format {model_format!r}; expected one of {_FORMATS}")


def model_hash(source: str, model_format: str) -> str:
    """sha256 over format + source with line endings normalized to LF."""
    norm = source.replace("\r\n", "\n").replace("\r", "\n")
    h = hashlib.sha256()
    h.update(model_format.encode())
    h.update(b"\0")
    h.update(norm.encode())
    return h.hexdigest()


# Antimony: `import "file"` at statement start (line start or after ';').
_ANT_IMPORT = re.compile(r"(?:^|[;\n\r{])\s*import\b", re.IGNORECASE)
_URL_OR_FILE = re.compile(r"(?:https?|ftp|file|jar)://|\bfile:", re.IGNORECASE)


def reject_unsafe(source: str, model_format: str) -> None:
    """Raise UnsafeModelError on imports or external model/file/URL references.

    Deliberately conservative: any URL/file scheme in SBML (other than XML
    namespace/schema declarations) is rejected.
    """
    _check_format(model_format)
    if model_format == "antimony":
        if _ANT_IMPORT.search(source):
            raise UnsafeModelError("Antimony 'import' statements are not allowed")
        if _URL_OR_FILE.search(source):
            raise UnsafeModelError("URL/file references are not allowed in Antimony source")
        return
    if re.search(r"externalModelDefinition", source, re.IGNORECASE):
        raise UnsafeModelError("SBML externalModelDefinition is not allowed")
    if re.search(r"<!ENTITY|<!DOCTYPE", source, re.IGNORECASE):
        raise UnsafeModelError("SBML DOCTYPE/ENTITY declarations are not allowed")
    # Strip xmlns declarations (legit URLs), then look for any remaining URL/file ref.
    stripped = re.sub(r"""xmlns(?::[\w.-]+)?\s*=\s*(?:"[^"]*"|'[^']*')""", "", source)
    stripped = re.sub(r"""xsi:schemaLocation\s*=\s*(?:"[^"]*"|'[^']*')""", "", stripped)
    if _URL_OR_FILE.search(stripped):
        raise UnsafeModelError("External file/URL references are not allowed in SBML")
    if re.search(r"""\b(?:comp:)?source\s*=""", stripped):
        raise UnsafeModelError("SBML 'source' attributes (external references) are not allowed")


def load_engine(source: str, model_format: str):
    """Compile the model with tellurium and return the RoadRunner instance."""
    _check_format(model_format)
    import tellurium as te

    if model_format == "antimony":
        return te.loada(source)
    return te.loadSBMLModel(source)


def _global_params(rr) -> list[str]:
    return list(rr.model.getGlobalParameterIds())


def _nonconstant_flags(rr) -> dict[str, bool]:
    """Map global parameter id -> SBML `constant` flag, read from the current SBML."""
    import libsbml

    m = libsbml.readSBMLFromString(rr.getCurrentSBML()).getModel()
    return {p.getId(): p.getConstant() for p in m.getListOfParameters()}


def _nonconstant_globals(rr) -> list[str]:
    # Assignment-rule / rate-rule targets are non-constant by SBML definition;
    # getCurrentSBML reports constant=False for them.
    flags = _nonconstant_flags(rr)
    return [p for p in _global_params(rr) if flags.get(p) is False]


def tunable_parameters(rr) -> list[str]:
    """Non-constant global parameters."""
    return _nonconstant_globals(rr)


def infer_observables(rr) -> dict[str, str]:
    """{id: id} for floating species + non-constant globals (incl. rule targets)."""
    ids: list[str] = []
    ids += list(rr.model.getFloatingSpeciesIds())
    ids += _nonconstant_globals(rr)
    # Assignment-rule targets (covers rules on species/compartments too).
    ids += [i for i in _rule_ids(rr) if i not in ids]
    return {i: i for i in dict.fromkeys(ids)}


def _rule_ids(rr) -> list[str]:
    import libsbml

    m = libsbml.readSBMLFromString(rr.getCurrentSBML()).getModel()
    return [r.getVariable() for r in m.getListOfRules() if r.isAssignment() and r.getVariable()]


def _known_ids(rr) -> set[str]:
    known = set(rr.model.getFloatingSpeciesIds())
    known |= set(rr.model.getBoundarySpeciesIds())
    known |= set(_global_params(rr))
    known |= set(rr.model.getCompartmentIds())
    known |= set(rr.model.getReactionIds())
    known |= set(_rule_ids(rr))
    return known


def validate_observables(rr, requested: list[str] | dict[str, str]) -> dict[str, str]:
    """Return {name: id}; raise UnknownObservableError for ids not in the model."""
    mapping = dict(requested) if isinstance(requested, dict) else {r: r for r in requested}
    known = _known_ids(rr)
    bad = sorted({v for v in mapping.values() if v not in known})
    if bad:
        raise UnknownObservableError(f"unknown observable id(s): {', '.join(bad)}")
    return mapping
