"""Phase-0 sanity tests — verify config and paths import cleanly."""

from __future__ import annotations

from pathlib import Path


def test_paths_module_provides_project_root() -> None:
    from src.paths import PROJECT_ROOT

    assert PROJECT_ROOT.is_absolute()
    assert (PROJECT_ROOT / "requirements.txt").exists()
    assert (PROJECT_ROOT / "app.py").exists()


def test_settings_loads_yaml_sections() -> None:
    from src.config import settings

    assert settings.commodity == "CORN"
    assert "IA" in settings.states
    assert settings.year_min == 2010
    assert settings.year_max == 2024
    assert settings.random_seed == 42


def test_logging_configures_once() -> None:
    from src.logging_setup import configure_logging

    configure_logging("INFO")
    configure_logging("INFO")  # idempotent


def test_root_app_importable() -> None:
    # The Streamlit import inside dashboard.main is lazy, so importing the root app
    # should succeed without streamlit installed.
    import importlib

    module = importlib.import_module("app")
    assert hasattr(module, "main"), "Root app.py must expose `main`"


def test_mlflow_uri_is_file_based_by_default() -> None:
    from src.config import settings

    uri = settings.mlflow_uri()
    assert uri.startswith("file://"), f"Expected file:// URI, got {uri}"
    assert "mlruns" in Path(uri.removeprefix("file://")).parts
