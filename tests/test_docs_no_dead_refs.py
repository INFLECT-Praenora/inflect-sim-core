"""Task 8: docs reflect the current surface — no references to deleted modules,
and the README names the ways to run.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# strings tied to the deleted twin service / old surrogate modules / old examples
DEAD = [
    "create_app", "ApiConfig", "TwinRegistry", "require_api",
    "surrogate_torch", "surrogate_curate", "surrogate_backend", "heavy_plane",
    "digital-twin API", "Digital-twin simulation API", "digital twins", "/twins",
    "digital_twin_demo", "vivarium_workflow_demo", "live_surrogate_demo",
    "docs/api.md", "docs/workflow-api.md",
]


def _docs():
    files = [REPO / "README.md"] + sorted((REPO / "docs").glob("*.md"))
    return {f: f.read_text(encoding="utf-8") for f in files if f.exists()}


def test_no_references_to_deleted_modules():
    offenders = {f.name: [d for d in DEAD if d in text] for f, text in _docs().items()}
    offenders = {name: hits for name, hits in offenders.items() if hits}
    assert not offenders, offenders


def test_deleted_docs_are_gone():
    assert not (REPO / "docs" / "api.md").exists()
    assert not (REPO / "docs" / "workflow-api.md").exists()


def test_readme_names_the_ways():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    for token in ("viva-sim", "serve", "sweep", "Nextflow", "connect", "catalog"):
        assert token in text, token
