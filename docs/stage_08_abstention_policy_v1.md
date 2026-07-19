# Stage 08 v3 — Train-only abstention policy

این مرحله مدل فریز شده Stage 07 را تغییر نمی دهد. فقط Policy بعد از مدل را
بازطراحی می کند تا سیستم بتواند در یک روز هیچ معامله ای انجام ندهد.

## ورودی ثابت

```text
Model: XGBoost
Feature set: I_full_40
Trial: 10
Raw features: 40
OOF rows: 51,840
Old baseline signals: 3,016
```

## Policy قدیمی

```text
Daily Top 5%
Minimum one signal per date
```

این Policy حتی در روزهای بسیار نامناسب بازار حداقل یک سیگنال صادر می کرد.

## Policy جدید

```text
Breadth regime gate
AND fixed raw-score threshold
AND daily Top-5% maximum cap
minimum signals per date = 0
```

ترتیب اجرا:

1. وضعیت Breadth روز بررسی می شود.
2. کاندیدهایی که از Gate عبور نمی کنند رد می شوند.
3. کاندیدهایی که امتیازشان از Threshold کمتر است رد می شوند.
4. از باقی مانده ها حداکثر Top 5% کل کاندیدهای همان روز انتخاب می شود.
5. اگر هیچ کاندیدی عبور نکند، خروجی روز صفر سیگنال است.

## Gateهای از قبل ثبت شده

```text
G0: همه Regimeها
G1: حذف broad_decline
G2: حذف broad_decline و deterioration
G3: فقط recovery_negative و recovery_positive
```

## Thresholdهای از قبل ثبت شده

Thresholdها از Quantile امتیازهای خام سیگنال های Baseline در OOF Train ساخته
می شوند:

```text
0%
25%
50%
65%
75%
```

این امتیازها Probability کالیبره شده نیستند و فقط Cutoff عملیاتی مدل فریز شده
هستند.

## Coverage

برای جلوگیری از انتخاب مصنوعی Policy با تعداد بسیار کم سیگنال:

```text
Pooled coverage >= 25% baseline
Every fold coverage >= 25% baseline fold
```

## اولویت انتخاب

```text
1. عبور از Coverage
2. کمترین False Positive
3. بیشترین Precision
4. بیشترین True Positive
5. بهترین Minimum Fold Precision
6. کمترین Fold Precision dispersion
7. بهترین Minimum Fold Specificity
8. بیشترین Signal Count
9. Gate ساده تر
10. Threshold پایین تر
```

## ممنوعیت ها

در Stage 08 موارد زیر استفاده نمی شوند:

```text
2021-2024 evaluation data
Unseen labels
Economic returns
Portfolio metrics
Model retraining
Hyperparameter changes
Feature changes
Calibration
```

## نصب

ZIP را در ریشه Repository و روی Branch زیر استخراج کنید:

```text
experiment/abstention-policy
```

Replace را تایید کنید، سپس:

```bash
python -m pytest -q tests/test_stage08_abstention_policy.py
rm -f CHECKSUMS_SHA256.txt
```

بعد کد را Commit کنید:

```bash
git add configs/signal_policy.yaml
git add notebooks/08_unseen_test_evaluation.ipynb
git add src/models/abstention_policy.py
git add tests/test_stage08_abstention_policy.py
git add docs/stage_08_abstention_policy_v1.md

git commit -m "feat: select train-only abstention policy in stage 08"
git push origin experiment/abstention-policy
```

سپس Notebook 08 را Restart Kernel و Run All کنید.

## خروجی های جدید

```text
results/manifests/08_abstention_signal_policy.json
results/manifests/08_abstention_policy_manifest.json

results/audits/08_abstention_oof_input_audit.csv
results/audits/08_abstention_baseline_policy_summary.csv
results/audits/08_abstention_baseline_fold_metrics.csv
results/audits/08_abstention_policy_candidate_grid.csv
results/audits/08_abstention_policy_all_fold_metrics.csv
results/audits/08_abstention_selected_policy_fold_metrics.csv
results/audits/08_abstention_selected_policy_date_audit.csv
results/audits/08_abstention_baseline_vs_selected.csv

results/predictions/08_abstention_oof_policy_predictions.csv
```

## خروجی نهایی مورد انتظار

مقادیر انتخاب شده Hardcode نشده اند. انتهای Notebook باید شامل این موارد باشد:

```text
Notebook 08 v3 integrity checks: PASSED
Selected model: xgboost
Selected feature set: I_full_40
Selected trial: 10
Selected policy ID: ...
Selected Breadth gate: ...
Minimum raw score: ...
Minimum signals per date: 0
Baseline signals: 3016
Selected signals: ...
Signal coverage: ...
Zero-signal dates: ...
True positives: ...
False positives: ...
Precision: ...
Specificity: ...
Sensitivity: ...
Minimum fold coverage: ...
Unseen test loaded: False
Economic returns used: False
Portfolio metrics used: False
```
