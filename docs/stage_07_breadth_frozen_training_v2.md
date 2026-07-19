# Stage 07 v2 — Frozen full-train breadth model

این بسته Notebook 07 قدیمی را با نسخه سازگار با تصمیم Stage 06 جایگزین می کند.

## مشکل نسخه قبلی

نسخه قبلی موارد زیر را Hardcode کرده بود:

- 35 فیچر؛
- 34 فیچر عددی؛
- فقط `gmma_state` به عنوان فیچر دسته ای؛
- Trial شماره 25؛
- چند Hyperparameter مشخص از اجرای قدیمی.

این فرض ها بعد از Stage 06 جدید معتبر نیستند.

## رفتار نسخه v2

Notebook جدید موارد زیر را مستقیما از خروجی فریز شده Stage 06 می خواند:

```text
primary_selected_model
primary_selected_feature_set
primary_selected_features
selected_trial_numbers
selected hyperparameters
```

در اجرای فعلی انتظار می رود:

```text
Model: xgboost
Feature set: I_full_40
Raw features: 40
Numeric features: 38
Categorical:
- gmma_state
- market_breadth_regime
```

شماره Trial و Hyperparameterها Hardcode نشده اند.

## کنترل ها

- جمعیت Train همان 118,464 کاندید است.
- 499 نماد استفاده می شود.
- Average Uniqueness روی کل Train و درون هر نماد محاسبه می شود.
- مدل Challenger آموزش داده نمی شود.
- Unseen Test خوانده نمی شود.
- Threshold انتخاب نمی شود.
- Calibration انجام نمی شود.
- فیلتر سخت started یا Breadth اعمال نمی شود.
- معیار عملکرد In-sample گزارش نمی شود.
- Pipeline کامل ذخیره و Reload equivalence کنترل می شود.

## نصب

ZIP را در ریشه Repository استخراج و Replace را تایید کنید.

سپس:

```bash
python -m pytest -q tests/test_stage07_breadth_frozen_training.py
git status --short
```

قبل از Run All، کد را Commit کنید:

```bash
git add notebooks/07_frozen_model_training.ipynb
git add configs/frozen_training.yaml
git add src/models/frozen_training.py
git add tests/test_stage07_breadth_frozen_training.py
git add docs/stage_07_breadth_frozen_training_v2.md

git commit -m "feat: align frozen model training with stage 06 breadth decision"
git push origin experiment/breadth-retrain
```

بعد Kernel را Restart و Notebook 07 را Run All کنید.

## خروجی نهایی مورد انتظار

```text
Notebook 07 integrity checks: PASSED
Selected primary model: xgboost
Selected Stage 06 feature set: I_full_40
Complete train candidate events: 118464
Train symbols: 499
Raw model features: 40
Numeric model features: 38
Categorical model features: ['gmma_state', 'market_breadth_regime']
Validation events used for fit: 0
Unseen-test events used for fit: 0
Training discrimination metrics reported: False
Calibration fitted: False
Threshold selected: False
Hard started filter applied: False
Hard breadth filter applied: False
Model reload equivalence: PASSED
```
