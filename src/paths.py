"""
Concept: ProjectPaths
Purpose: Provide a single source of truth for filesystem locations used across the project.
State: Absolute paths derived from PROJECT_ROOT (the parent of `src/`).
Actions: Expose path constants importable from anywhere.
Operational principle: Importing `PROJECT_ROOT` from any module returns the same absolute path,
regardless of the current working directory.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Top-level directories
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_INTERIM: Path = DATA_DIR / "interim"
DATA_PROCESSED: Path = DATA_DIR / "processed"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
MODELS_DIR: Path = PROJECT_ROOT / "models"
MLRUNS_DIR: Path = PROJECT_ROOT / "mlruns"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"
TESTS_DIR: Path = PROJECT_ROOT / "tests"

# Key artifacts
CONFIG_YAML: Path = CONFIG_DIR / "config.yaml"
FEATURES_PARQUET: Path = DATA_PROCESSED / "features.parquet"
COUNTIES_GEOJSON: Path = DATA_PROCESSED / "counties_geo.json"
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.pkl"


def ensure_dirs() -> None:
    """Create all standard directories if they do not yet exist."""
    for path in (
        DATA_RAW,
        DATA_INTERIM,
        DATA_PROCESSED,
        DOCS_DIR,
        MODELS_DIR,
        MLRUNS_DIR,
        NOTEBOOKS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        SCRIPTS_DIR,
        TESTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
