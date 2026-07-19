# Stage 08 v2 — Train-only policy lock

نسخه قبلی Notebook 08 فقط با `model_name=xgboost` فیلتر می کرد. بعد از
Stage 06 جدید، فایل OOF شامل XGBoost برای 9 Feature Set است؛ بنابراین فیلتر
فقط بر اساس نام مدل باعث تکرار event_id و چند برابر شدن جمعیت OOF می شود.

نسخه v2 فقط این Variant را مصرف می کند:

```text
model_name = xgboost
feature_set_name = I_full_40
```

## تفاوت روش شناختی مهم

Stage 06 قبلا Policy روزانه Top 5% را در انتخاب مدل و Feature Set استفاده کرده
است. در نتیجه Stage 08 نباید دوباره:

- روش Calibration انتخاب کند؛
- Fraction سیگنال جستجو کند؛
- Threshold احتمالی انتخاب کند.

Stage 08 فقط Policy انتخاب شده را دقیقا بازسازی و فریز می کند.

## هویت مورد انتظار

```text
OOF rows: 51,840
Signals: 3,016
True positives: 1,987
False positives: 1,029
Precision: 0.6588196286472149
```

Policy:

```text
Top 5% per date
minimum 1 signal
ceil quota
score descending
symbol ascending
event_id ascending
```

## نصب و اجرا

ZIP را در ریشه Repository استخراج و Replace را تایید کنید.

```bash
python -m pytest -q tests/test_stage08_policy_lock.py
rm -f CHECKSUMS_SHA256.txt
```

قبل از Run All:

```bash
git add notebooks/08_unseen_test_evaluation.ipynb
git add configs/signal_policy.yaml
git add tests/test_stage08_policy_lock.py
git add docs/stage_08_policy_lock_v2.md

git commit -m "feat: freeze exact stage 06 model feature policy in stage 08"
git push origin experiment/breadth-retrain
```

سپس Kernel را Restart و Notebook 08 را Run All کنید.

## خروجی نهایی مورد انتظار

```text
Notebook 08 integrity checks: PASSED
Selected model: xgboost
Selected feature set: I_full_40
Raw feature count: 40
Score policy: raw identity ranking
Probability calibrator fitted: False
Selected signal policy: daily top 5%
Minimum signals per date: 1
Total selected OOF signals: 3016
True positives: 1987
False positives: 1029
Precision: 0.6588196286472149
Policy reselected in Stage 08: False
Unseen-test loaded: False
Unseen-test labels used: False
```
