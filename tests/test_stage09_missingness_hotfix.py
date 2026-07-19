from __future__ import annotations

import json
from pathlib import Path


def _notebook_source() -> str:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (
            root
            / "notebooks"
            / "09_signal_level_evaluation.ipynb"
        ).read_text(encoding="utf-8")
    )
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_strict_all_feature_notna_assertion_removed() -> None:
    source = _notebook_source()
    assert (
        "assert candidate_panel[SELECTED_FEATURES].notna().all().all()"
        not in source
    )


def test_stage04_added_features_still_must_be_complete() -> None:
    source = _notebook_source()
    assert (
        "list(STAGE04_BREADTH_FEATURES)"
        in source
    )
    assert (
        "09_unseen_test_candidate_missingness_audit.csv"
        in source
    )
    assert "frozen_pipeline_imputation_used" in source


def test_candidate_rows_are_not_dropped_or_manually_filled() -> None:
    source = _notebook_source()
    assert "do not drop candidate rows" in source
    assert "do not fill values manually" in source
    assert "missing_by_feature" in source
    assert "missing_by_feature.lt" in source


def test_numeric_and_categorical_roles_are_normalized() -> None:
    source = _notebook_source()
    assert "for feature in SELECTED_NUMERIC_FEATURES" in source
    assert "for feature in SELECTED_CATEGORICAL_FEATURES" in source
    assert 'astype("string")' in source


def test_all_code_cells_compile() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (
            root
            / "notebooks"
            / "09_signal_level_evaluation.ipynb"
        ).read_text(encoding="utf-8")
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        compile(
            "".join(cell.get("source", [])),
            f"cell_{index}",
            "exec",
        )
