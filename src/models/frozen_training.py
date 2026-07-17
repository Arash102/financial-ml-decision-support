"""Utilities for fitting and serializing the frozen full-train model."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

FROZEN_TRAINING_SCHEMA_VERSION = (
    "stage07_v1_full_train_average_uniqueness_xgboost"
)


def weighted_scale_pos_weight(
    y: Iterable[int],
    sample_weight: Iterable[float],
    *,
    mode: str,
) -> float:
    y_array = np.asarray(list(y), dtype=int)
    weight_array = np.asarray(list(sample_weight), dtype=float)
    if y_array.ndim != 1 or weight_array.ndim != 1:
        raise ValueError("y and sample_weight must be one-dimensional.")
    if len(y_array) == 0 or len(y_array) != len(weight_array):
        raise ValueError("y and sample_weight must have equal nonzero length.")
    if not np.isfinite(weight_array).all() or (weight_array <= 0.0).any():
        raise ValueError("sample_weight must contain finite positive values.")
    if mode == "none":
        return 1.0
    if mode != "fold_weighted_ratio":
        raise ValueError(f"Unknown class_weight_mode: {mode}")
    positive_mass = float(weight_array[y_array == 1].sum())
    negative_mass = float(weight_array[y_array == 0].sum())
    if positive_mass <= 0.0 or negative_mass <= 0.0:
        raise ValueError("Both weighted classes are required.")
    return negative_mass / positive_mass


def dataframe_fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"Fingerprint columns are missing: {missing}")
    selected = frame.loc[:, columns].copy()
    row_hashes = pd.util.hash_pandas_object(
        selected,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    metadata = json.dumps(
        {
            "columns": columns,
            "dtypes": [str(selected[column].dtype) for column in columns],
            "rows": int(len(selected)),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def fitted_feature_names(pipeline: Pipeline) -> list[str]:
    if "preprocess" not in pipeline.named_steps:
        raise KeyError("Pipeline does not contain a preprocess step.")
    return [
        str(value)
        for value in pipeline.named_steps["preprocess"].get_feature_names_out()
    ]


def save_pipeline_with_reload_check(
    pipeline: Pipeline,
    path: Path,
    X_probe: pd.DataFrame,
    *,
    compression: int = 3,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    probability_before = pipeline.predict_proba(X_probe)[:, 1]
    joblib.dump(pipeline, path, compress=int(compression))
    loaded = joblib.load(path)
    probability_after = loaded.predict_proba(X_probe)[:, 1]
    if probability_before.shape != probability_after.shape:
        raise AssertionError("Reloaded model changed prediction shape.")
    maximum_absolute_difference = float(
        np.max(np.abs(probability_before - probability_after))
    )
    if not np.allclose(
        probability_before,
        probability_after,
        atol=1.0e-12,
        rtol=1.0e-12,
    ):
        raise AssertionError("Reloaded pipeline probabilities changed.")
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return {
        "model_path": str(path),
        "model_size_bytes": int(path.stat().st_size),
        "model_sha256": digest.hexdigest(),
        "probe_rows": int(len(X_probe)),
        "reload_probability_max_abs_difference": maximum_absolute_difference,
        "reload_equivalence_passed": True,
    }
