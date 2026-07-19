from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.models.abstention_policy import (
    candidate_symbol_from_path,
    load_candidate_regime_sources,
)


def test_candidate_symbol_is_reconstructed_from_filename() -> None:
    path = Path("ABC_train_candidates.csv")
    assert candidate_symbol_from_path(path) == "ABC"


def test_candidate_regime_loader_does_not_require_event_id(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ABC_train_candidates.csv"
    pd.DataFrame(
        {
            "dEven": [
                "2020-01-01",
                "2020-01-02",
            ],
            "market_breadth_regime": [
                "recovery_positive",
                "deterioration",
            ],
        }
    ).to_csv(path, index=False)

    regimes, errors = load_candidate_regime_sources([path])

    assert errors.empty
    assert regimes["event_id"].tolist() == [
        "ABC::2020-01-01",
        "ABC::2020-01-02",
    ]
    assert regimes["symbol"].tolist() == ["ABC", "ABC"]
    assert regimes["market_breadth_regime"].tolist() == [
        "recovery_positive",
        "deterioration",
    ]


def test_invalid_candidate_date_is_audited(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ABC_train_candidates.csv"
    pd.DataFrame(
        {
            "dEven": ["invalid-date"],
            "market_breadth_regime": [
                "recovery_positive",
            ],
        }
    ).to_csv(path, index=False)

    regimes, errors = load_candidate_regime_sources([path])

    assert regimes.empty
    assert len(errors) == 1
    assert errors.iloc[0]["error_type"] == "ValueError"


def test_notebook_uses_stage06_event_identity_rule() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (
            root
            / "notebooks"
            / "08_unseen_test_evaluation.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assert "load_candidate_regime_sources" in source
    assert "symbol::YYYY-MM-DD" in source
    assert 'required = {\n            "event_id"' not in source
    assert 'usecols=[\n                    "event_id"' not in source
    assert "missing_regime_event_ids" in source


def test_all_notebook_code_cells_compile() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (
            root
            / "notebooks"
            / "08_unseen_test_evaluation.ipynb"
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
