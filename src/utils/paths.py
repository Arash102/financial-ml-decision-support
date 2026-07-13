"""Repository and data-path helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def find_repository_root(start: Path | None = None) -> Path:
    """Find the repository root by searching upward for project markers."""
    current = (start or Path.cwd()).resolve()

    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
        if (candidate / "configs").exists() and (candidate / "notebooks").exists():
            return candidate

    raise FileNotFoundError(
        "Repository root was not found. Open the project folder in VS Code "
        "and run the notebook from inside that folder."
    )


def resolve_data_root(
    paths_config: dict[str, Any],
    repository_root: Path,
) -> Path:
    """
    Resolve the data root.

    Priority:
    1. FINML_DATA_ROOT environment variable, when defined.
    2. Repository root, when data_root_mode is 'repository_root'.
    3. Explicit data_root_default from configuration.
    """
    env_name = str(paths_config.get("environment_variable", "FINML_DATA_ROOT"))
    env_value = os.getenv(env_name)

    if env_value:
        return Path(env_value).expanduser().resolve()

    mode = str(paths_config.get("data_root_mode", "")).strip().lower()
    if mode == "repository_root":
        return repository_root.resolve()

    default_value = paths_config.get("data_root_default")
    if default_value is None:
        raise ValueError(
            "paths.yaml must define either data_root_mode: repository_root "
            "or data_root_default."
        )

    return Path(str(default_value)).expanduser().resolve()


def repository_result_paths(
    repository_root: Path,
    paths_config: dict[str, Any],
) -> dict[str, Path]:
    """Resolve repository-managed result directories."""
    result: dict[str, Path] = {}

    for key, relative_value in paths_config["repository_results"].items():
        path = repository_root / str(relative_value)
        path.mkdir(parents=True, exist_ok=True)
        result[key] = path

    return result


def data_paths(
    data_root: Path,
    paths_config: dict[str, Any],
) -> dict[str, Path]:
    """Resolve local data directories."""
    result: dict[str, Path] = {}

    for key, relative_value in paths_config["directories"].items():
        path = data_root / str(relative_value)
        path.mkdir(parents=True, exist_ok=True)
        result[key] = path

    return result
