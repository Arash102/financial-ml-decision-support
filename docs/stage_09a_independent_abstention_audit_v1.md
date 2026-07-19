# Stage 09A — ممیزی مستقل Abstention Policy

این مرحله هیچ مدل یا Policy جدیدی انتخاب نمی کند. هدف آن کنترل مستقل خروجی
Stage 09 است.

## مواردی که مستقل بازسازی می شوند

```text
1. SHA فایل Policy و Outcome-free inference lock
2. هویت تمام 78,189 کاندید
3. هویت تمام سیگنال های انتخاب شده
4. Breadth gate
5. Raw-score threshold
6. Daily Top-5% maximum quota
7. امکان صفر سیگنال در روز
8. Precision، Specificity و Sensitivity
9. Win rate، Payoff ratio و Profit factor رویدادی
10. Corrected outcome تمام سیگنال ها مستقیما از raw_data
```

کد ممیزی عمدا این ماژول ها را Import نمی کند:

```text
src.models.abstention_policy
src.evaluation.unseen_test_signal
```

در نتیجه Policy و Outcomeها با پیاده سازی مستقل کنترل می شوند.

## ورودی های لازم

```text
results/manifests/08_abstention_signal_policy.json
results/manifests/08_abstention_policy_manifest.json
results/manifests/09_abstention_inference_lock.json
results/manifests/09_abstention_signal_evaluation_manifest.json

results/predictions/09_abstention_inference_lock.csv
results/predictions/09_abstention_signal_evaluation.csv
results/predictions/09_abstention_selected_signals.csv

raw_data/*.csv
```

سه فایل Prediction باید روی سیستم محلی موجود باشند؛ لازم نیست وارد Git شوند.

## نصب

ZIP را در ریشه Repository و روی Branch زیر استخراج کنید:

```text
experiment/abstention-policy
```

فایل های بسته:

```text
scripts/audit_stage09_abstention.py
src/audit/stage09_abstention_audit.py
src/audit/__init__.py
tests/test_stage09a_independent_audit.py
docs/stage_09a_independent_abstention_audit_v1.md
```

تست:

```bash
python -m pytest -q tests/test_stage09a_independent_audit.py
rm -f CHECKSUMS_SHA256.txt
```

کد را قبل از اجرای ممیزی ثبت کنید:

```bash
git add scripts/audit_stage09_abstention.py
git add src/audit/stage09_abstention_audit.py
git add src/audit/__init__.py
git add tests/test_stage09a_independent_audit.py
git add docs/stage_09a_independent_abstention_audit_v1.md

git commit -m "test: add independent stage 09 abstention audit"
git push origin experiment/abstention-policy
```

سپس ممیزی را اجرا کنید:

```bash
python scripts/audit_stage09_abstention.py
```

## خروجی های ممیزی

```text
results/manifests/09a_independent_abstention_audit_manifest.json

results/audits/09a_independent_abstention_audit_checks.csv
results/audits/09a_independent_abstention_audit_summary.csv
results/audits/09a_independent_abstention_audit_report_fa.md
results/audits/09a_independent_classification_metric_comparison.csv
results/audits/09a_independent_outcome_metric_comparison.csv
results/audits/09a_independent_outcome_reconstruction_audit.csv
results/audits/09a_independent_outcome_reconstruction_errors.csv
results/audits/09a_independent_corrected_outcomes_by_year.csv
results/audits/09a_independent_corrected_outcomes_by_regime.csv
```

## نتیجه مورد انتظار

```text
Stage 09A independent audit: PASSED
Failed checks: 0
Stage 08 policy ID: G3_recovery_only__q0000
Candidate events: 78189
Selected signals: 1815
Dates with signal: 375
Zero-signal dates: 445
True positives: 1023
False positives: 792
Precision: 0.563636...
Specificity: 0.981715...
Sensitivity: 0.029335...
Corrected win rate: 0.563636...
Corrected payoff ratio: 2.449871...
Corrected profit factor: 3.168418...
Raw selected outcomes reconstructed: 1815
Raw reconstruction errors: 0
```

این ممیزی صحت فنی را بررسی می کند. نتیجه 2021 تا 2024 همچنان Post-hoc است و
Profit Factor رویدادی نباید به عنوان Profit Factor پرتفوی قابل معامله گزارش
شود.
