# Stage 09 v5 — Frozen abstention-policy post-hoc retest

این مرحله Policy انتخاب شده در Stage 08 را بدون هیچ تغییر روی دوره تاریخی
2021 تا 2024 اعمال می کند.

## ورودی های فریز شده

```text
Model: XGBoost
Feature set: I_full_40
Trial: 10
Model SHA256:
73a781551cdabd6bb67f5b9c0836a8683372e46487d9aade29e6320006086c1f
```

Policy:

```text
Policy ID: G3_recovery_only__q0000
Allowed regimes:
- recovery_negative
- recovery_positive

Minimum raw score:
0.3844631016254425

Daily maximum fraction:
0.05

Minimum signals per date:
0
```

تمام مقادیر Policy از فایل فریز شده زیر خوانده می شوند:

```text
results/manifests/08_abstention_signal_policy.json
```

Stage 09 هیچ Gate، Threshold یا Fraction جدیدی انتخاب نمی کند.

## ترتیب علمی اجرا

```text
1. ساخت علّی همان 40 ویژگی
2. امتیازدهی با مدل فریز شده Stage 07
3. اعمال دقیق Policy فریز شده Stage 08
4. ثبت outcome-free inference lock
5. محاسبه SHA قفل
6. سپس خواندن Label و Outcome
7. بازسازی corrected selected-signal outcomes
8. گزارش Post-hoc بدون تغییر Policy
```

تابع جدید:

```text
apply_abstention_policy_inference
```

برای اعمال Policy به Label یا Outcome نیاز ندارد. این جداسازی مانع استفاده
ناخواسته از نتیجه آینده پیش از قفل شدن سیگنال ها می شود.

## تفاوت با Stage 09 v4

نسخه قبلی در تمام 820 تاریخ حداقل یک سیگنال و روزانه Top 5% صادر می کرد.

نسخه جدید:

```text
Recovery gate
AND raw-score cutoff
AND Top-5% maximum cap
```

را اعمال می کند و ممکن است در یک روز صفر سیگنال داشته باشد.

تعداد سیگنال دوره 2021 تا 2024 از قبل تعیین یا Hardcode نشده است. همچنین هیچ
مقدار عملکردی مانند Precision، Win Rate، Payoff یا Profit Factor پیش از قفل
به Notebook داده نشده است.

## وضعیت علمی

دوره 2021 تا 2024 قبلا در توسعه پروژه بررسی شده است. بنابراین نتیجه این
Notebook:

```text
Post-hoc diagnostic retest
```

است و تایید مستقل محسوب نمی شود.

## خروجی های اصلی

```text
results/predictions/09_abstention_inference_lock.csv
results/predictions/09_abstention_signal_evaluation.csv
results/predictions/09_abstention_selected_signals.csv

results/manifests/09_abstention_inference_lock.json
results/manifests/09_abstention_signal_evaluation_manifest.json
```

Auditها با پیشوند زیر ساخته می شوند:

```text
results/audits/09_abstention_
```

گزارش ها شامل موارد زیر هستند:

```text
Signal coverage vs old Top-5%-minimum-one policy
Dates with signal
Zero-signal dates
Precision
Specificity
Sensitivity
Yearly corrected outcomes
Barrier outcomes
Breadth-regime outcomes
```

## نصب

ZIP را در ریشه Repository روی Branch زیر استخراج کنید:

```text
experiment/abstention-policy
```

Replace را تایید کنید و سپس:

```bash
python -m pytest -q tests/test_stage09_abstention_posthoc.py
rm -f CHECKSUMS_SHA256.txt
```

کد را قبل از اجرای Notebook ثبت کنید:

```bash
git add configs/unseen_test_evaluation.yaml
git add notebooks/09_signal_level_evaluation.ipynb
git add src/models/abstention_policy.py
git add tests/test_stage09_abstention_posthoc.py
git add docs/stage_09_abstention_posthoc_v5.md

git commit -m "feat: apply frozen abstention policy in stage 09"
git push origin experiment/abstention-policy
```

سپس:

```text
Restart Kernel
Run All
```

## خروجی نهایی مورد انتظار

اعداد عملکردی عمدا از قبل مشخص نشده اند. انتهای Notebook باید ساختاری شبیه
زیر داشته باشد:

```text
Notebook 09 v5 integrity checks: PASSED
Scientific status: post-hoc retest on previously inspected period
Confirmatory claim allowed: False
Stage 08 policy ID: G3_recovery_only__q0000
Breadth gate: G3_recovery_only
Allowed regimes: ['recovery_negative', 'recovery_positive']
Minimum raw score: 0.3844631016254425
Daily maximum fraction: 0.05
Minimum signals per date: 0
Long candidate events: 78189
Signal dates: 820
Old Top-5%-minimum-one signals: 4311
Frozen abstention-policy signals: ...
Signal coverage vs old policy: ...
Dates with signal: ...
Zero-signal dates: ...
Signal precision: ...
Signal specificity: ...
Signal sensitivity: ...
Corrected selected-signal win rate: ...
Corrected selected-signal payoff ratio: ...
Corrected selected-signal profit factor: ...
Expected selected-signal count supplied before lock: False
Expected performance values supplied before lock: False
Policy reselected in Stage 09: False
Portfolio backtest performed: False
```
