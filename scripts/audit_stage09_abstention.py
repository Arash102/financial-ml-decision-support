from __future__ import annotations

from pathlib import Path
import json
import sys
import traceback

import numpy as np
import pandas as pd

from src.audit.stage09_abstention_audit import (
    AUDIT_SCHEMA_VERSION,
    FrozenPolicy,
    as_bool,
    classification_metrics,
    compare_metric_dicts,
    corrected_outcome_metrics,
    file_sha256,
    independently_reconstruct_selected_outcomes,
    load_json,
    load_yaml,
    locate_repository_root,
    reconstruct_policy_decisions,
    resolve_raw_data_root,
    values_close,
)


def add_check(
    rows: list[dict[str, object]],
    *,
    name: str,
    passed: bool,
    observed: object,
    expected: object,
    category: str,
    details: str = "",
) -> None:
    rows.append({
        "category": category,
        "check_name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
        "details": details,
    })


def metric_row(
    metrics: dict[str, object],
    prefix: str,
) -> dict[str, object]:
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
    }


def main() -> int:
    repository_root = locate_repository_root(Path.cwd())
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

    paths_config = load_yaml(
        repository_root / "configs" / "paths.yaml"
    )
    stage09_config = load_yaml(
        repository_root
        / "configs"
        / "unseen_test_evaluation.yaml"
    )

    result_root = repository_root / "results"
    audit_root = result_root / "audits"
    manifest_root = result_root / "manifests"
    prediction_root = result_root / "predictions"
    audit_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    policy_path = (
        manifest_root
        / "08_abstention_signal_policy.json"
    )
    stage08_manifest_path = (
        manifest_root
        / "08_abstention_policy_manifest.json"
    )
    lock_manifest_path = (
        manifest_root
        / "09_abstention_inference_lock.json"
    )
    stage09_manifest_path = (
        manifest_root
        / "09_abstention_signal_evaluation_manifest.json"
    )
    lock_path = (
        prediction_root
        / "09_abstention_inference_lock.csv"
    )
    full_evaluation_path = (
        prediction_root
        / "09_abstention_signal_evaluation.csv"
    )
    selected_path = (
        prediction_root
        / "09_abstention_selected_signals.csv"
    )
    stage09_classification_audit_path = (
        audit_root
        / "09_abstention_signal_classification_metrics.csv"
    )
    stage09_outcome_audit_path = (
        audit_root
        / "09_abstention_signal_outcome_summary.csv"
    )

    required_paths = [
        policy_path,
        stage08_manifest_path,
        lock_manifest_path,
        stage09_manifest_path,
        lock_path,
        full_evaluation_path,
        selected_path,
        stage09_classification_audit_path,
        stage09_outcome_audit_path,
    ]
    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Required Stage 09 audit inputs are missing: "
            + ", ".join(missing_paths)
        )

    policy_artifact = load_json(policy_path)
    stage08_manifest = load_json(stage08_manifest_path)
    lock_manifest = load_json(lock_manifest_path)
    stage09_manifest = load_json(stage09_manifest_path)

    policy = FrozenPolicy(
        gate_name=str(
            policy_artifact["market_gate"]["gate_name"]
        ),
        allowed_regimes=tuple(
            str(value)
            for value in policy_artifact[
                "market_gate"
            ]["allowed_regimes"]
        ),
        minimum_raw_score=float(
            policy_artifact[
                "score_threshold"
            ]["minimum_raw_score"]
        ),
        maximum_daily_fraction=float(
            policy_artifact[
                "daily_cap"
            ]["maximum_fraction"]
        ),
        minimum_signals_per_date=int(
            policy_artifact[
                "daily_cap"
            ]["minimum_signals_per_date"]
        ),
    )
    policy.validate()

    lock = pd.read_csv(lock_path, low_memory=False)
    evaluation = pd.read_csv(
        full_evaluation_path,
        low_memory=False,
    )
    selected = pd.read_csv(
        selected_path,
        low_memory=False,
    )

    for frame in [lock, evaluation, selected]:
        frame["event_id"] = frame["event_id"].astype(str)
        frame["symbol"] = frame["symbol"].astype(str)
        frame["dEven"] = pd.to_datetime(
            frame["dEven"],
            errors="raise",
        ).dt.normalize()

    lock["selected_signal"] = as_bool(
        lock["selected_signal"]
    )
    evaluation["selected_signal"] = as_bool(
        evaluation["selected_signal"]
    )
    selected["selected_signal"] = as_bool(
        selected["selected_signal"]
    )

    checks: list[dict[str, object]] = []

    # Artifact identity and SHA checks.
    actual_policy_sha = file_sha256(policy_path)
    actual_lock_sha = file_sha256(lock_path)
    add_check(
        checks,
        name="stage08_policy_file_sha256",
        passed=(
            actual_policy_sha
            == str(lock_manifest["stage08_policy_file_sha256"])
            == str(
                stage09_manifest["lineage"][
                    "stage08_policy_file_sha256"
                ]
            )
        ),
        observed=actual_policy_sha,
        expected=lock_manifest["stage08_policy_file_sha256"],
        category="lineage",
    )
    add_check(
        checks,
        name="inference_lock_sha256",
        passed=(
            actual_lock_sha
            == str(lock_manifest["inference_lock_file_sha256"])
            == str(
                stage09_manifest["inference_lock"][
                    "inference_lock_file_sha256"
                ]
            )
        ),
        observed=actual_lock_sha,
        expected=lock_manifest["inference_lock_file_sha256"],
        category="lineage",
    )
    add_check(
        checks,
        name="stage08_policy_configuration_hash",
        passed=(
            str(policy_artifact["configuration_hash"])
            == str(
                stage08_manifest[
                    "selected_policy_configuration_hash"
                ]
            )
            == str(
                lock_manifest[
                    "stage08_policy_configuration_hash"
                ]
            )
            == str(
                stage09_manifest["lineage"][
                    "stage08_policy_configuration_hash"
                ]
            )
        ),
        observed=policy_artifact["configuration_hash"],
        expected=stage08_manifest[
            "selected_policy_configuration_hash"
        ],
        category="lineage",
    )
    add_check(
        checks,
        name="stage08_policy_id",
        passed=(
            str(stage08_manifest["selected_policy_id"])
            == str(lock_manifest["stage08_policy_id"])
            == str(
                stage09_manifest["lineage"][
                    "stage08_policy_id"
                ]
            )
        ),
        observed=lock_manifest["stage08_policy_id"],
        expected=stage08_manifest["selected_policy_id"],
        category="lineage",
    )

    forbidden_lock_columns = {
        "meta_label",
        "label",
        "event_return",
        "original_event_return",
        "barrier_touched",
        "event_end_date",
        "corrected_event_return",
        "corrected_winner",
        "holding_period_observations",
    }
    observed_forbidden = sorted(
        forbidden_lock_columns.intersection(lock.columns)
    )
    add_check(
        checks,
        name="outcome_free_lock_schema",
        passed=not observed_forbidden,
        observed="|".join(observed_forbidden),
        expected="",
        category="lock",
        details="No label or outcome columns may exist in the lock.",
    )

    add_check(
        checks,
        name="unique_lock_event_ids",
        passed=lock["event_id"].is_unique,
        observed=int(lock["event_id"].duplicated().sum()),
        expected=0,
        category="identity",
    )
    add_check(
        checks,
        name="unique_evaluation_event_ids",
        passed=evaluation["event_id"].is_unique,
        observed=int(
            evaluation["event_id"].duplicated().sum()
        ),
        expected=0,
        category="identity",
    )
    add_check(
        checks,
        name="unique_selected_event_ids",
        passed=selected["event_id"].is_unique,
        observed=int(
            selected["event_id"].duplicated().sum()
        ),
        expected=0,
        category="identity",
    )

    lock_ids = set(lock["event_id"])
    evaluation_ids = set(evaluation["event_id"])
    selected_ids = set(selected["event_id"])
    lock_selected_ids = set(
        lock.loc[lock["selected_signal"], "event_id"]
    )
    evaluation_selected_ids = set(
        evaluation.loc[
            evaluation["selected_signal"],
            "event_id",
        ]
    )
    add_check(
        checks,
        name="lock_vs_evaluation_candidate_identity",
        passed=lock_ids == evaluation_ids,
        observed=len(lock_ids.symmetric_difference(evaluation_ids)),
        expected=0,
        category="identity",
    )
    add_check(
        checks,
        name="selected_identity_across_three_files",
        passed=(
            selected_ids
            == lock_selected_ids
            == evaluation_selected_ids
        ),
        observed=len(
            selected_ids.symmetric_difference(
                lock_selected_ids
            )
            | selected_ids.symmetric_difference(
                evaluation_selected_ids
            )
        ),
        expected=0,
        category="identity",
    )

    # Independent policy reconstruction.
    reconstructed = reconstruct_policy_decisions(
        lock,
        policy=policy,
    )
    lock_by_id = lock.set_index("event_id", drop=False)
    audit_by_id = reconstructed.set_index(
        "event_id",
        drop=False,
    )
    aligned = lock_by_id.join(
        audit_by_id[
            [
                "audit_daily_candidate_count",
                "audit_daily_maximum_quota",
                "audit_market_gate_pass",
                "audit_score_threshold_pass",
                "audit_policy_eligible",
                "audit_daily_eligible_count",
                "audit_daily_signal_quota",
                "audit_daily_eligible_rank",
                "audit_daily_selected_score_cutoff",
                "audit_selected_signal",
            ]
        ],
        how="inner",
        validate="one_to_one",
    )

    boolean_pairs = [
        ("market_gate_pass", "audit_market_gate_pass"),
        ("score_threshold_pass", "audit_score_threshold_pass"),
        ("policy_eligible", "audit_policy_eligible"),
        ("selected_signal", "audit_selected_signal"),
    ]
    for stored_column, audit_column in boolean_pairs:
        stored_values = as_bool(aligned[stored_column])
        audit_values = as_bool(aligned[audit_column])
        mismatches = int((stored_values != audit_values).sum())
        add_check(
            checks,
            name=f"independent_{stored_column}",
            passed=mismatches == 0,
            observed=mismatches,
            expected=0,
            category="policy_reconstruction",
        )

    integer_pairs = [
        (
            "daily_candidate_count",
            "audit_daily_candidate_count",
        ),
        (
            "daily_maximum_quota",
            "audit_daily_maximum_quota",
        ),
        (
            "daily_eligible_count",
            "audit_daily_eligible_count",
        ),
        (
            "daily_signal_quota",
            "audit_daily_signal_quota",
        ),
        (
            "daily_eligible_rank",
            "audit_daily_eligible_rank",
        ),
    ]
    for stored_column, audit_column in integer_pairs:
        stored_values = pd.to_numeric(
            aligned[stored_column],
            errors="raise",
        ).astype(int)
        audit_values = pd.to_numeric(
            aligned[audit_column],
            errors="raise",
        ).astype(int)
        mismatches = int((stored_values != audit_values).sum())
        add_check(
            checks,
            name=f"independent_{stored_column}",
            passed=mismatches == 0,
            observed=mismatches,
            expected=0,
            category="policy_reconstruction",
        )

    stored_cutoff = pd.to_numeric(
        aligned["daily_selected_score_cutoff"],
        errors="coerce",
    ).to_numpy(dtype=float)
    audit_cutoff = pd.to_numeric(
        aligned["audit_daily_selected_score_cutoff"],
        errors="coerce",
    ).to_numpy(dtype=float)
    cutoff_equal = np.isclose(
        stored_cutoff,
        audit_cutoff,
        rtol=1.0e-12,
        atol=1.0e-12,
        equal_nan=True,
    )
    add_check(
        checks,
        name="independent_daily_selected_score_cutoff",
        passed=bool(cutoff_equal.all()),
        observed=int((~cutoff_equal).sum()),
        expected=0,
        category="policy_reconstruction",
    )

    dates = int(lock["dEven"].nunique())
    selected_count = int(lock["selected_signal"].sum())
    dates_with_signal = int(
        lock.loc[
            lock["selected_signal"],
            "dEven",
        ].nunique()
    )
    zero_signal_dates = int(dates - dates_with_signal)
    old_top5_signals = int(
        lock.groupby("dEven", observed=False)[
            "old_top5_minimum_one_quota"
        ].first().sum()
    )
    coverage = float(selected_count / old_top5_signals)

    structural_observed = {
        "candidate_events": int(len(lock)),
        "signal_dates": dates,
        "selected_signals": selected_count,
        "dates_with_signal": dates_with_signal,
        "zero_signal_dates": zero_signal_dates,
        "old_top5_minimum_one_signals": old_top5_signals,
        "signal_coverage_vs_old_top5": coverage,
    }
    structural_expected = {
        "candidate_events": lock_manifest["candidate_events"],
        "signal_dates": lock_manifest["signal_dates"],
        "selected_signals": lock_manifest["selected_signals"],
        "dates_with_signal": lock_manifest["dates_with_signal"],
        "zero_signal_dates": lock_manifest["zero_signal_dates"],
        "old_top5_minimum_one_signals": (
            lock_manifest["old_top5_minimum_one_signals"]
        ),
        "signal_coverage_vs_old_top5": (
            lock_manifest["signal_coverage_vs_old_top5"]
        ),
    }
    for comparison in compare_metric_dicts(
        structural_observed,
        structural_expected,
        tolerance=1.0e-12,
    ):
        add_check(
            checks,
            name=f"structural_{comparison['metric']}",
            passed=bool(comparison["passed"]),
            observed=comparison["observed"],
            expected=comparison["expected"],
            category="structure",
        )

    # Policy values must match the pre-run config and the final manifest.
    config_policy = stage09_config["frozen_signal_policy"]
    policy_value_checks = {
        "gate_name": (
            policy.gate_name,
            config_policy["expected_gate_name"],
        ),
        "allowed_regimes": (
            list(policy.allowed_regimes),
            list(config_policy["expected_allowed_regimes"]),
        ),
        "minimum_raw_score": (
            policy.minimum_raw_score,
            config_policy["expected_minimum_raw_score"],
        ),
        "maximum_daily_fraction": (
            policy.maximum_daily_fraction,
            config_policy["expected_maximum_daily_fraction"],
        ),
        "minimum_signals_per_date": (
            policy.minimum_signals_per_date,
            config_policy["expected_minimum_signals_per_date"],
        ),
    }
    for name, (observed, expected) in policy_value_checks.items():
        passed = (
            observed == expected
            if isinstance(observed, list)
            else values_close(
                observed,
                expected,
                tolerance=1.0e-12,
            )
        )
        add_check(
            checks,
            name=f"frozen_policy_{name}",
            passed=passed,
            observed=observed,
            expected=expected,
            category="policy_lineage",
        )

    # Independent signal-classification metrics.
    classification = classification_metrics(
        evaluation["meta_label"],
        evaluation["selected_signal"],
    )
    stored_classification = (
        pd.read_csv(
            stage09_classification_audit_path,
            low_memory=False,
        )
        .iloc[0]
        .to_dict()
    )
    manifest_classification = stage09_manifest[
        "signal_classification_metrics"
    ]
    classification_comparison_rows = []
    for comparison in compare_metric_dicts(
        classification,
        stored_classification,
        tolerance=1.0e-12,
    ):
        classification_comparison_rows.append({
            "source": "stage09_audit_csv",
            **comparison,
        })
        add_check(
            checks,
            name=f"classification_csv_{comparison['metric']}",
            passed=bool(comparison["passed"]),
            observed=comparison["observed"],
            expected=comparison["expected"],
            category="classification",
        )
    for comparison in compare_metric_dicts(
        classification,
        manifest_classification,
        tolerance=1.0e-12,
    ):
        classification_comparison_rows.append({
            "source": "stage09_manifest",
            **comparison,
        })
        add_check(
            checks,
            name=f"classification_manifest_{comparison['metric']}",
            passed=bool(comparison["passed"]),
            observed=comparison["observed"],
            expected=comparison["expected"],
            category="classification",
        )

    # Independent corrected-outcome metrics.
    outcome_metrics = corrected_outcome_metrics(selected)
    stored_outcome = (
        pd.read_csv(
            stage09_outcome_audit_path,
            low_memory=False,
        )
        .iloc[0]
        .to_dict()
    )
    manifest_outcome = stage09_manifest[
        "selected_signal_corrected_outcomes"
    ]
    outcome_comparison_rows = []
    for comparison in compare_metric_dicts(
        outcome_metrics,
        stored_outcome,
        tolerance=1.0e-12,
    ):
        outcome_comparison_rows.append({
            "source": "stage09_audit_csv",
            **comparison,
        })
        add_check(
            checks,
            name=f"outcome_csv_{comparison['metric']}",
            passed=bool(comparison["passed"]),
            observed=comparison["observed"],
            expected=comparison["expected"],
            category="outcome_metrics",
        )
    for comparison in compare_metric_dicts(
        outcome_metrics,
        manifest_outcome,
        tolerance=1.0e-12,
    ):
        outcome_comparison_rows.append({
            "source": "stage09_manifest",
            **comparison,
        })
        add_check(
            checks,
            name=f"outcome_manifest_{comparison['metric']}",
            passed=bool(comparison["passed"]),
            observed=comparison["observed"],
            expected=comparison["expected"],
            category="outcome_metrics",
        )

    # Independent raw-price reconstruction for all selected signals.
    raw_root = resolve_raw_data_root(
        repository_root,
        paths_config,
    )
    reconstruction_audit, reconstruction_errors = (
        independently_reconstruct_selected_outcomes(
            selected,
            raw_root=raw_root,
            signal_generation_end=pd.Timestamp(
                stage09_config["temporal_scope"][
                    "signal_generation_end"
                ]
            ),
            tail_end=pd.Timestamp(
                stage09_config["temporal_scope"][
                    "outcome_observation_tail_end"
                ]
            ),
            horizon=int(
                stage09_config[
                    "corrected_event_outcome_policy"
                ]["horizon_observations"]
            ),
            upper_barrier=float(
                stage09_config[
                    "corrected_event_outcome_policy"
                ]["upper_barrier_return"]
            ),
            lower_barrier=float(
                stage09_config[
                    "corrected_event_outcome_policy"
                ]["lower_barrier_return"]
            ),
        )
    )

    reconstruction_audit_path = (
        audit_root
        / "09a_independent_outcome_reconstruction_audit.csv"
    )
    reconstruction_error_path = (
        audit_root
        / "09a_independent_outcome_reconstruction_errors.csv"
    )
    reconstruction_audit.to_csv(
        reconstruction_audit_path,
        index=False,
        encoding="utf-8-sig",
    )
    reconstruction_errors.to_csv(
        reconstruction_error_path,
        index=False,
        encoding="utf-8-sig",
    )

    add_check(
        checks,
        name="raw_outcome_reconstruction_event_count",
        passed=(
            len(reconstruction_audit) == len(selected)
            and reconstruction_errors.empty
        ),
        observed=len(reconstruction_audit),
        expected=len(selected),
        category="raw_outcome_reconstruction",
    )
    add_check(
        checks,
        name="raw_outcome_reconstruction_errors",
        passed=reconstruction_errors.empty,
        observed=len(reconstruction_errors),
        expected=0,
        category="raw_outcome_reconstruction",
    )
    add_check(
        checks,
        name="raw_outcome_reconstruction_all_rows_pass",
        passed=(
            not reconstruction_audit.empty
            and reconstruction_audit["passed"].all()
        ),
        observed=int(
            (~reconstruction_audit["passed"]).sum()
        ) if not reconstruction_audit.empty else len(selected),
        expected=0,
        category="raw_outcome_reconstruction",
    )

    # Year and regime metrics are independently recomputed for diagnostics.
    year_rows: list[dict[str, object]] = []
    selected["calendar_year"] = selected[
        "dEven"
    ].dt.year.astype(int)
    for year, group in selected.groupby(
        "calendar_year",
        sort=True,
        observed=False,
    ):
        year_rows.append({
            "calendar_year": int(year),
            **corrected_outcome_metrics(group),
        })
    by_year = pd.DataFrame(year_rows)
    by_year.to_csv(
        audit_root
        / "09a_independent_corrected_outcomes_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )

    regime_rows: list[dict[str, object]] = []
    for regime, group in selected.groupby(
        "market_breadth_regime",
        sort=True,
        observed=False,
    ):
        regime_rows.append({
            "market_breadth_regime": str(regime),
            **corrected_outcome_metrics(group),
        })
    by_regime = pd.DataFrame(regime_rows)
    by_regime.to_csv(
        audit_root
        / "09a_independent_corrected_outcomes_by_regime.csv",
        index=False,
        encoding="utf-8-sig",
    )

    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(
        audit_root
        / "09a_independent_abstention_audit_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(classification_comparison_rows).to_csv(
        audit_root
        / "09a_independent_classification_metric_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(outcome_comparison_rows).to_csv(
        audit_root
        / "09a_independent_outcome_metric_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    failed_checks = check_frame.loc[
        ~check_frame["passed"].astype(bool)
    ].copy()
    audit_passed = bool(failed_checks.empty)

    summary = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_passed": audit_passed,
        "checks": int(len(check_frame)),
        "failed_checks": int(len(failed_checks)),
        "stage08_policy_id": stage08_manifest["selected_policy_id"],
        "stage08_policy_file_sha256": actual_policy_sha,
        "stage08_policy_configuration_hash": (
            policy_artifact["configuration_hash"]
        ),
        "inference_lock_sha256": actual_lock_sha,
        "candidate_events": int(len(lock)),
        "signal_dates": dates,
        "selected_signals": selected_count,
        "old_top5_minimum_one_signals": old_top5_signals,
        "signal_coverage_vs_old_top5": coverage,
        "dates_with_signal": dates_with_signal,
        "zero_signal_dates": zero_signal_dates,
        **metric_row(classification, "classification"),
        **metric_row(outcome_metrics, "corrected_outcome"),
        "raw_outcome_reconstruction_rows": int(
            len(reconstruction_audit)
        ),
        "raw_outcome_reconstruction_errors": int(
            len(reconstruction_errors)
        ),
        "scientific_status": (
            "post_hoc_retest_previously_inspected_period"
        ),
        "confirmatory_claim_allowed": False,
        "portfolio_backtest_performed": False,
        "transaction_costs_applied": False,
    }
    summary_frame = pd.DataFrame([summary])
    summary_frame.to_csv(
        audit_root
        / "09a_independent_abstention_audit_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report_lines = [
        "# گزارش ممیزی مستقل Stage 09A",
        "",
        f"- وضعیت ممیزی: {'PASSED' if audit_passed else 'FAILED'}",
        f"- تعداد کنترل ها: {len(check_frame)}",
        f"- کنترل های ناموفق: {len(failed_checks)}",
        f"- SHA قفل استنتاج: `{actual_lock_sha}`",
        f"- تعداد کاندیدها: {len(lock)}",
        f"- تعداد سیگنال ها: {selected_count}",
        f"- روزهای دارای سیگنال: {dates_with_signal}",
        f"- روزهای بدون سیگنال: {zero_signal_dates}",
        f"- False Positive: {classification['false_positive']}",
        f"- Precision: {classification['precision']:.12f}",
        f"- Specificity: {classification['specificity']:.12f}",
        f"- Sensitivity: {classification['sensitivity']:.12f}",
        f"- Win rate رویدادی: {outcome_metrics['win_rate']:.12f}",
        f"- Payoff ratio رویدادی: {outcome_metrics['payoff_ratio']:.12f}",
        f"- Profit factor رویدادی: {outcome_metrics['profit_factor']:.12f}",
        (
            "- بازسازی مستقل Outcome از فایل های خام: "
            f"{len(reconstruction_audit)} رویداد، "
            f"{len(reconstruction_errors)} خطا"
        ),
        "",
        "این ممیزی فقط صحت فنی Policy و Outcomeهای رویدادی را تایید می کند.",
        "نتیجه 2021 تا 2024 همچنان Post-hoc است و بازده رویدادی، بازده پرتفوی قابل اجرا نیست.",
    ]
    if not failed_checks.empty:
        report_lines.extend([
            "",
            "## کنترل های ناموفق",
            "",
        ])
        for row in failed_checks.itertuples(index=False):
            report_lines.append(
                f"- `{row.check_name}`: observed={row.observed}, "
                f"expected={row.expected}"
            )

    report_path = (
        audit_root
        / "09a_independent_abstention_audit_report_fa.md"
    )
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    audit_manifest = {
        "stage": "09A",
        "status": (
            "independent_audit_passed"
            if audit_passed
            else "independent_audit_failed"
        ),
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "inputs": {
            "stage08_policy": str(policy_path),
            "stage08_policy_sha256": actual_policy_sha,
            "stage09_inference_lock": str(lock_path),
            "stage09_inference_lock_sha256": actual_lock_sha,
            "stage09_full_evaluation": str(
                full_evaluation_path
            ),
            "stage09_selected_signals": str(selected_path),
            "raw_data_root": str(raw_root),
        },
        "independence": {
            "stage08_policy_module_imported": False,
            "stage09_outcome_helper_imported": False,
            "policy_decisions_reconstructed_independently": True,
            "classification_metrics_recomputed_independently": True,
            "corrected_outcome_metrics_recomputed_independently": True,
            "selected_outcomes_reconstructed_from_raw_prices": True,
        },
        "summary": summary,
        "outputs": {
            "checks": str(
                audit_root
                / "09a_independent_abstention_audit_checks.csv"
            ),
            "summary": str(
                audit_root
                / "09a_independent_abstention_audit_summary.csv"
            ),
            "report": str(report_path),
            "raw_reconstruction_audit": str(
                reconstruction_audit_path
            ),
            "raw_reconstruction_errors": str(
                reconstruction_error_path
            ),
        },
    }
    manifest_path = (
        manifest_root
        / "09a_independent_abstention_audit_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            audit_manifest,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        "Stage 09A independent audit:",
        "PASSED" if audit_passed else "FAILED",
    )
    print("Checks:", len(check_frame))
    print("Failed checks:", len(failed_checks))
    print("Stage 08 policy ID:", stage08_manifest["selected_policy_id"])
    print("Inference lock SHA256:", actual_lock_sha)
    print("Candidate events:", len(lock))
    print("Selected signals:", selected_count)
    print("Dates with signal:", dates_with_signal)
    print("Zero-signal dates:", zero_signal_dates)
    print("True positives:", classification["true_positive"])
    print("False positives:", classification["false_positive"])
    print("Precision:", classification["precision"])
    print("Specificity:", classification["specificity"])
    print("Sensitivity:", classification["sensitivity"])
    print("Corrected win rate:", outcome_metrics["win_rate"])
    print("Corrected payoff ratio:", outcome_metrics["payoff_ratio"])
    print("Corrected profit factor:", outcome_metrics["profit_factor"])
    print(
        "Raw selected outcomes reconstructed:",
        len(reconstruction_audit),
    )
    print(
        "Raw reconstruction errors:",
        len(reconstruction_errors),
    )
    print("Report:", report_path)
    print("Manifest:", manifest_path)

    return 0 if audit_passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
