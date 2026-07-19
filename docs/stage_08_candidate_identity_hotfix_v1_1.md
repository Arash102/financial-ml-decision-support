# Stage 08 v1.1 — Candidate identity hotfix

## علت خطا

فایل های زیر:

```text
data_ready/candidates/train/*_train_candidates.csv
```

ستون `event_id` را ذخیره نمی کنند. در Stage 06 شناسه رویداد هنگام خواندن
فایل ساخته شده بود:

```text
symbol = نام فایل قبل از _train_candidates.csv
event_id = symbol + "::" + YYYY-MM-DD(dEven)
```

نسخه اولیه Stage 08 اشتباها انتظار داشت `event_id` داخل CSV وجود داشته باشد؛
به همین دلیل تمام یا بخش بزرگی از فایل ها در
`08_abstention_candidate_source_errors.csv` ثبت شدند.

## اصلاح

نسخه جدید دقیقا همان قرارداد هویتی Stage 06 را بازسازی می کند:

1. نماد از نام فایل استخراج می شود.
2. `dEven` به تاریخ نرمال تبدیل می شود.
3. `event_id` به شکل `symbol::YYYY-MM-DD` ساخته می شود.
4. `market_breadth_regime` با OOFها بر اساس همین شناسه Join می شود.
5. نبود حتی یک OOF event در منبع Regime باعث توقف اجرا می شود.

هیچ ردیفی حذف نمی شود و هیچ Join تقریبی بر اساس تاریخ تنها انجام نمی شود.

## نصب

ZIP را در ریشه Repository استخراج و Replace کنید:

```bash
python -m pytest -q tests/test_stage08_candidate_identity_hotfix.py
rm -f CHECKSUMS_SHA256.txt
```

سپس:

```bash
git add notebooks/08_unseen_test_evaluation.ipynb
git add src/models/abstention_policy.py
git add tests/test_stage08_candidate_identity_hotfix.py
git add docs/stage_08_candidate_identity_hotfix_v1_1.md

git commit -m "fix: reconstruct stage 06 candidate event identity in stage 08"
git push origin experiment/abstention-policy
```

بعد Notebook 08 را Restart Kernel و Run All کنید.
