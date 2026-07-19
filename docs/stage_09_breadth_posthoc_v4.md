# Stage 09 v4 — Breadth model post-hoc retest

این بسته Notebook 09 قدیمی را با نسخه سازگار با مدل جدید جایگزین می کند.

## اصلاح علمی مهم

بازه 2021 تا 2024 در نسخه های قبلی پروژه دیده شده و طراحی Breadth نیز از
تحلیل های همین بازه تاثیر گرفته است. بنابراین اجرای جدید:

```text
Post-hoc retest
```

است و نباید به عنوان Confirmatory Unseen Test یا External Validation معرفی شود.

تایید واقعی باید با داده های بعدی که در طراحی استفاده نشده اند یا Paper
Trading آینده انجام شود.

## مشکل Notebook قبلی

نسخه قدیمی این موارد را قفل کرده بود:

```text
35 raw features
37 transformed features
old model SHA256
old inference-lock SHA256
old expected performance metrics
```

این موارد با مدل جدید `XGBoost + I_full_40` ناسازگارند.

## نسخه جدید

```text
Selected model: xgboost
Selected feature set: I_full_40
Selected trial: 10
Raw features: 40
Numeric features: 38
Categorical features: 2
Transformed features: 47
```

پنج فیچر اضافه شده دقیقا مطابق Stage 04 بازسازی می شوند:

```text
started_run_length
market_breadth_raw
market_breadth_ema30
market_breadth_slope5
market_breadth_regime
```

## قواعد Breadth

```text
Breadth = (N_positive - N_negative) / N_valid
```

- بازده صفر در مخرج باقی می ماند؛
- نماد فاقد مشاهده معتبر در آن تاریخ از مخرج حذف می شود؛
- بازده هر نماد نسبت به مشاهده معتبر قبلی همان نماد محاسبه می شود؛
- EMA30 با `adjust=False` و `min_periods=30`؛
- Slope5 برابر EMA30 امروز منهای EMA30 پنج تاریخ بازار قبل؛
- هیچ فیلتر سخت Breadth یا started اعمال نمی شود.

## قفل قبل از Outcome

قبل از خواندن Label و Outcome:

```text
candidate features
raw XGBoost ranking scores
daily ranks
Top-5% signal selection
```

در فایل Outcome-free lock ذخیره می شوند.

هیچ Expected Lock Hash یا Expected Performance Metric از اجرای قبلی در
نسخه جدید قرار ندارد.

## جمعیت های ثابت ساختاری

به دلیل ثابت بودن Rule کاندید و Quota:

```text
Raw evaluation rows: 371,321
Eligible events: 359,454
Long candidate events: 78,189
Signal dates: 820
Selected signals: 4,311
```

این اعداد معیار عملکرد نیستند.

## نصب

ZIP را در ریشه Repository استخراج و Replace را تایید کنید.

```bash
python -m pytest -q tests/test_stage09_breadth_posthoc.py
rm -f CHECKSUMS_SHA256.txt
```

خروجی تست:

```text
8 passed
```

قبل از Run All:

```bash
git add notebooks/09_signal_level_evaluation.ipynb
git add configs/unseen_test_evaluation.yaml
git add src/features/unseen_breadth.py
git add tests/test_stage09_breadth_posthoc.py
git add docs/stage_09_breadth_posthoc_v4.md

git commit -m "feat: add truthful breadth post-hoc retest in stage 09"
git push origin experiment/breadth-retrain
```

سپس Kernel را Restart و Notebook 09 را Run All کنید.

## خروجی نهایی

مقادیر عملکرد از قبل Hardcode نشده اند. در انتها باید این کنترل ها پاس شوند:

```text
Notebook 09 v4 integrity checks: PASSED
Scientific status: post-hoc retest on previously inspected period
Confirmatory claim allowed: False
Selected model: xgboost
Selected feature set: I_full_40
Selected trial: 10
Raw model features: 40
Transformed model features: 47
Long candidate events: 78189
Signal dates: 820
Frozen selected signals: 4311
Scores locked before outcomes: True
Expected performance values supplied before lock: False
Portfolio backtest performed: False
Transaction costs applied: False
```

Precision، Specificity، Win Rate، Payoff و Profit Factor برای مدل جدید فقط
پس از اجرا مشخص می شوند.
