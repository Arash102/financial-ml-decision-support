"""Feature-role and structural leakage helpers."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def build_feature_role_table(columns_config: dict[str, Any]) -> pd.DataFrame:
    """Build the configured structural role for every candidate feature."""
    candidate_features = list(columns_config["candidate_features"])
    policy = columns_config["feature_policy"]

    context_only = set(policy["context_only_price_features"])
    legacy_zigzag = set(policy["legacy_zigzag_columns"])
    base_model = set(policy["base_model_candidates"])

    rows: list[dict[str, object]] = []

    for feature in candidate_features:
        memberships = {
            "context_only_price": feature in context_only,
            "legacy_zigzag": feature in legacy_zigzag,
            "base_model_candidate": feature in base_model,
        }
        membership_count = sum(bool(value) for value in memberships.values())

        if membership_count != 1:
            role = "configuration_error"
            structurally_approved = False
            reason = (
                "Feature must belong to exactly one structural role; "
                f"memberships={memberships}"
            )
        elif memberships["context_only_price"]:
            role = "context_only"
            structurally_approved = False
            reason = (
                "Adjusted OHLC level retained for labeling, audit, and "
                "causal reconstruction; excluded from direct model inputs."
            )
        elif memberships["legacy_zigzag"]:
            role = "legacy_zigzag_rejected"
            structurally_approved = False
            reason = (
                "Legacy new_2 ZigZag columns are replaced by a directly "
                "confirmation-gated causal reconstruction in Stage 04."
            )
        else:
            role = "model_candidate"
            structurally_approved = True
            reason = "Eligible for train-only data-quality approval."

        rows.append(
            {
                "feature": feature,
                "structural_role": role,
                "structurally_approved": structurally_approved,
                "structural_reason": reason,
                "role_membership_count": membership_count,
            }
        )

    return pd.DataFrame(rows)


def audit_prohibited_feature_names(
    features: list[str],
    patterns: list[str],
) -> pd.DataFrame:
    """Audit feature names against explicit target/future-derived patterns."""
    rows: list[dict[str, object]] = []

    for feature in features:
        matched_patterns = [
            pattern
            for pattern in patterns
            if re.search(pattern, feature, flags=re.IGNORECASE)
        ]
        rows.append(
            {
                "feature": feature,
                "prohibited_name_hit": bool(matched_patterns),
                "matched_patterns": "|".join(matched_patterns),
            }
        )

    return pd.DataFrame(rows)


def finalize_feature_approval(
    role_table: pd.DataFrame,
    quality_table: pd.DataFrame,
    *,
    maximum_missing_fraction: float,
    reject_all_missing: bool,
    reject_constant: bool,
) -> pd.DataFrame:
    """Combine structural roles with train-only quality checks."""
    merged = role_table.merge(
        quality_table,
        on="feature",
        how="left",
        validate="one_to_one",
    )

    approved: list[bool] = []
    reasons: list[str] = []

    for row in merged.itertuples(index=False):
        row_reasons: list[str] = []

        if not bool(row.structurally_approved):
            row_reasons.append(str(row.structural_role))

        if reject_all_missing and int(row.finite_values) == 0:
            row_reasons.append("all_missing_or_nonfinite")

        if float(row.missing_fraction) > maximum_missing_fraction:
            row_reasons.append("missing_fraction_above_limit")

        is_constant = bool(row.is_constant_on_finite_train_values)
        if reject_constant and is_constant:
            row_reasons.append("constant_on_train")

        is_approved = (
            bool(row.structurally_approved)
            and not row_reasons
        )

        approved.append(is_approved)
        reasons.append(
            "approved" if is_approved else "|".join(row_reasons)
        )

    merged["approved_for_modeling"] = approved
    merged["approval_reason"] = reasons
    return merged
