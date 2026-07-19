# Stage 09 v4.1 missingness hotfix

## علت خطا

Assertion قبلی همه 40 فیچر را مجبور می کرد که هیچ مقدار گمشده ای نداشته
باشند:

```python
assert candidate_panel[SELECTED_FEATURES].notna().all().all()
```

این شرط با Pipeline فریز شده Stage 07 سازگار نبود. Pipeline آموزش دیده:

- فیچرهای عددی را با Median Imputer؛
- فیچرهای دسته ای را با Most-Frequent Imputer

مدیریت می کند. بنابراین وجود بعضی Missingهای علّی در فیچرهای پایه مجاز است و
نباید باعث حذف کاندید، پر کردن دستی یا تغییر جمعیت شود.

## سیاست اصلاح شده

- پنج فیچر جدید Stage 04 باید کاملا موجود باشند.
- هیچ فیچر انتخاب شده ای نباید برای تمام کاندیدها Missing باشد.
- مقدار Infinity ممنوع است.
- Missingهای فیچرهای پایه به Pipeline فریز شده سپرده می شوند.
- تعداد و هویت کاندیدها تغییر نمی کند.
- فایل زیر برای ممیزی نوشته می شود:

```text
results/audits/09_unseen_test_candidate_missingness_audit.csv
```

## نصب

ZIP را در ریشه Repository استخراج و Replace را تایید کنید.

```bash
python -m pytest -q tests/test_stage09_missingness_hotfix.py
rm -f CHECKSUMS_SHA256.txt
```

سپس:

```bash
git add notebooks/09_signal_level_evaluation.ipynb
git add tests/test_stage09_missingness_hotfix.py
git add docs/stage_09_missingness_hotfix_v4_1.md

git commit -m "fix: align stage 09 missingness checks with frozen pipeline"
git push origin experiment/breadth-retrain
```

بعد Kernel را Restart و Notebook 09 را Run All کنید.
