# Stage 04 causal breadth extension v2

این بسته جایگزین نسخه قبلی و معیوب است.

روش این نسخه:

- فایل `src/features/preprocessing.py` را تغییر نمی دهد.
- داخل سلول های موجود Notebook 04 جستجو و جایگزینی انجام نمی دهد.
- فقط دو سلول جدید به انتهای Notebook 04 اضافه می کند.
- بعد از اجرای کامل Stage 04 اصلی، کاندیداهای Train را بدون حذف هیچ ردیفی با Breadth و started غنی می کند.
- تمام فایل های کاندیدا ابتدا در پوشه موقت ساخته و ممیزی می شوند و فقط پس از موفقیت کامل جایگزین می شوند.
- هویت و ترتیب ردیف های کاندیدا کنترل می شود.
- هیچ فیلتر سخت `started` یا Breadth اعمال نمی شود.

## فیچرهای افزوده شده

- `started_run_length`
- `market_breadth_raw`
- `market_breadth_ema30`
- `market_breadth_slope5`
- `market_breadth_regime`

## نصب

بسته را در ریشه ریپازیتوری Extract کنید و سپس اجرا کنید:

```bash
python append_stage04_breadth_extension.py
python -m pytest -q tests/test_stage04_breadth_extension.py
git status --short
```

پس از موفقیت تست ها، فایل نصب کننده و Backup موقت را حذف کنید:

```bash
rm -f append_stage04_breadth_extension.py
rm -f CHECKSUMS_SHA256.txt
rm -f notebooks/04_feature_and_leakage_audit.ipynb.stage04-breadth-backup
```

فایل های دائمی:

```text
notebooks/04_feature_and_leakage_audit.ipynb
src/features/stage04_breadth_extension.py
tests/test_stage04_breadth_extension.py
docs/stage_04_breadth_extension_v2.md
```

سپس Commit:

```bash
git add notebooks/04_feature_and_leakage_audit.ipynb
git add src/features/stage04_breadth_extension.py
git add tests/test_stage04_breadth_extension.py
git add docs/stage_04_breadth_extension_v2.md

git commit -m "feat: add transactional causal breadth extension to stage 04"
git push origin experiment/breadth-retrain
```

بعد Kernel را Restart کنید و فقط Notebook 04 را Run All کنید.

در سلول پایانی باید این موارد دیده شوند:

```text
Stage 04 breadth extension: PASSED
Final approved model features: 40
Candidate identity preserved: True
Hard started filter applied: False
Hard breadth filter applied: False
```


## Hotfix v2.1: causal warm-up encoding

The exact EMA30 and slope5 definitions are unchanged after sufficient causal
history exists. During the initial warm-up only, unavailable numeric values are
encoded as `0.0`, while `market_breadth_regime` remains
`warmup_unavailable`. The audit records how many rows were encoded. No future
data is used and no candidate row is removed.
