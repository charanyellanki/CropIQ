"""
Concept: Settings
Purpose: Provide validated, immutable runtime configuration for CropIQ.
State: Project metadata, geography, time splits, paths, API endpoints, and model hyperparameters
    loaded from `config/config.yaml` and overlaid with environment variables (.env loaded at
    import time).
Actions: `load_settings()`, `Settings.get_yaml_section(name)`, attribute access.
Operational principle: Importing `settings` from any module returns the same validated singleton
    with the merged YAML + env configuration.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.paths import CONFIG_YAML, PROJECT_ROOT

# Load .env BEFORE any Settings instantiation — see gotcha #14.
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read the YAML config file and return it as a dict.

    Args:
        path: Absolute path to the YAML file.

    Returns:
        Parsed YAML as a nested dict. Returns an empty dict if the file is missing or empty.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at top level of {path}, got {type(data).__name__}")
    return data


class Settings(BaseSettings):
    """Top-level runtime configuration for CropIQ.

    Environment variables (e.g., `NASS_API_KEY`) take precedence over YAML values.
    Use `get_yaml_section(name)` to access nested YAML config (e.g., model hyperparameters).
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Secrets / credentials ---
    nass_api_key: str = Field(default="", description="USDA NASS Quick Stats API key.")
    noaa_token: str = Field(default="", description="NOAA CDO API token (fallback only).")

    # --- Optional overrides ---
    mlflow_tracking_uri: str | None = Field(default=None)
    log_level: str = Field(default="INFO")

    # --- YAML-backed config (populated in __init__) ---
    _yaml: dict[str, Any] = {}

    def __init__(self, **values: Any) -> None:
        super().__init__(**values)
        # Bypass pydantic immutability check (Settings stores all fields above).
        object.__setattr__(self, "_yaml", _load_yaml(CONFIG_YAML))

    # --- Convenience accessors ---
    def get_yaml_section(self, name: str) -> dict[str, Any]:
        """Return the named top-level section of `config.yaml`."""
        section = self._yaml.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"YAML section '{name}' is not a mapping (got {type(section).__name__}).")
        return section

    @property
    def states(self) -> list[str]:
        return list(self.get_yaml_section("geography").get("states", []))

    @property
    def state_fips(self) -> dict[str, str]:
        return dict(self.get_yaml_section("geography").get("state_fips", {}))

    @property
    def year_min(self) -> int:
        return int(self.get_yaml_section("time").get("year_min", 2010))

    @property
    def year_max(self) -> int:
        return int(self.get_yaml_section("time").get("year_max", 2024))

    @property
    def train_year_max(self) -> int:
        return int(self.get_yaml_section("time").get("train_year_max", 2020))

    @property
    def val_years(self) -> list[int]:
        return list(self.get_yaml_section("time").get("val_years", [2021, 2022]))

    @property
    def test_years(self) -> list[int]:
        return list(self.get_yaml_section("time").get("test_years", [2023, 2024]))

    @property
    def random_seed(self) -> int:
        return int(self.get_yaml_section("project").get("random_seed", 42))

    @property
    def commodity(self) -> str:
        return str(self.get_yaml_section("target").get("commodity", "CORN"))

    def mlflow_uri(self) -> str:
        """Return the effective MLflow tracking URI."""
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        env_override = os.environ.get("MLFLOW_TRACKING_URI")
        if env_override:
            return env_override
        return f"file://{PROJECT_ROOT / 'mlruns'}"


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Return the singleton `Settings` instance.

    Returns:
        The validated `Settings` object. Subsequent calls return the same instance.
    """
    return Settings()


# Module-level singleton — import as `from src.config import settings`.
settings: Settings = load_settings()
