# Stage 06 v4 — Policy-first Optuna and Breadth feature-set selection

این بسته جایگزین مستقیم Notebook 06 فعلی است.

## چرا Stage 06 تغییر کرد؟

Stage 04 اکنون 40 فیچر دارد:

- 35 فیچر پایه؛
- `started_run_length`؛
- سه فیچر عددی Breadth؛
- یک فیچر دسته ای Breadth.

نسخه قدیمی Notebook 06 تعداد 35 فیچر، 34 فیچر عددی و فقط
`gmma_state` را فرض می کرد. همچنین انتخاب Trial و مدل فقط بر ROC AUC
متمرکز بود.

نسخه v4 اولویت عملیاتی پروژه را اعمال می کند:

1. کمترین False Positive؛
2. بیشترین Precision؛
3. بیشترین پایداری Specificity بین Foldها؛
4. Average Precision و ROC AUC به عنوان معیارهای تکمیلی.

## طراحی محاسباتی

برای جلوگیری از 9 برابر شدن Optuna:

- RF و XGBoost فقط یک بار روی مجموعه کامل 40 فیچر Tune می شوند؛
- هر Trial همه 5 Fold را اجرا می کند؛
- Objective مورد استفاده TPE همان میانگین ROC AUC است؛
- انتخاب نهایی Trial با Policy ثابت Top 5% و اولویت False Positive انجام می شود؛
- سپس Hyperparameterهای منتخب هر خانواده ثابت می شوند؛
- 9 Feature Set با Hyperparameterهای مشترک مقایسه می شوند.

این طراحی باعث می شود مقایسه Feature Setها از تغییر همزمان Hyperparameterها
آلوده نشود و هزینه محاسباتی کنترل شود.

## Feature Setها

- `A_baseline_35`
- `B_plus_started`
- `C_plus_breadth_raw`
- `D_plus_breadth_ema30`
- `E_plus_breadth_slope5`
- `F_plus_all_continuous_breadth`
- `G_plus_breadth_regime`
- `H_plus_slope5_and_regime`
- `I_full_40`

هیچ فیلتر سخت `started` یا Breadth اعمال نمی شود.

## Policy ثابت

برای هر تاریخ Validation:

```text
quota = max(1, ceil(candidate_count × 0.05))
```

ترتیب انتخاب:

```text
probability_positive descending
symbol ascending
event_id ascending
```

هیچ Probability Threshold انتخاب نمی شود.

## فایل های بسته

```text
notebooks/06_optuna_model_selection.ipynb
src/models/policy_selection.py
configs/stage06_breadth_retrain.yaml
tests/test_stage06_policy_selection.py
docs/stage_06_breadth_policy_v1.md
```

## نصب

ZIP را در ریشه Repository استخراج و Replace را تایید کنید.

سپس:

```bash
python -m pytest -q tests/test_stage06_policy_selection.py
git status --short
```

قبل از Run All، فایل ها را Commit کنید:

```bash
git add notebooks/06_optuna_model_selection.ipynb
git add src/models/policy_selection.py
git add configs/stage06_breadth_retrain.yaml
git add tests/test_stage06_policy_selection.py
git add docs/stage_06_breadth_policy_v1.md

git commit -m "feat: add policy-first breadth model selection to stage 06"
git push origin experiment/breadth-retrain
```

بعد Kernel را Restart و Notebook 06 را Run All کنید.

## خروجی نهایی مورد انتظار

```text
Notebook 06 integrity checks: PASSED
Candidate events: 118464
Final pooled-model features: 40
Pre-registered feature sets: 9
Frozen Stage 05 folds: 5
COMPLETE trials per tuned model: 30
Optuna sampler objective: equal-fold mean ROC AUC
Trial selection priority: minimum false positives under fixed Top 5% policy
Final selection priority: minimum false positives, then precision and fold stability
Threshold selection performed: False
Hard started filter applied: False
Hard breadth filter applied: False
Unseen-test used in Stage 06 decisions: False
```

اجرای کامل سنگین تر از Notebook 06 قبلی است، چون بعد از 60 Trial،
دو خانواده مدل روی 9 Feature Set و 5 Fold نیز ارزیابی می شوند.
