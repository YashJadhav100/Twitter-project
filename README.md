# Twitter-like JSONL Data Starter

This mini-starter helps you **quickly load, sanity-check, and convert** the dataset the professor sent you.

## Folder
- `process_jsonl.py` — loads JSONL (one JSON object per line), validates key fields, parses dates, dedupes, and writes Parquet + a small CSV sample
- `schema.json` — expected fields you can edit
- `requirements.txt` — install these once in your virtual env
- `run_example.sh` / `run_example.bat` — quick-run helpers

## 1) Setup (one time)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Put the file the professor sent
Copy it to this folder and note its filename. If the file is large, you can test on a capped number of rows first.

## 3) Run
Example for a file named `all_im1_11082024.json`:
```bash
python process_jsonl.py --input all_im1_11082024.json --schema schema.json --outdir out --rows 5000
```

**Flags:**
- `--rows` is optional; remove it to read all rows.
- `--sample 2000` changes how many rows go into `out/sample.csv`.

## 4) Results
- `out/data.parquet` — columnar, fast to load
- `out/sample.csv` — quick peek of cleaned rows
- `out/SUMMARY.md` — tiny quality report (nulls, langs, etc.)

## 5) Next steps (suggested)
- Inspect `SUMMARY.md`
- Load `data.parquet` in notebooks for EDA and modeling
- If your fields differ, edit `schema.json` (add/remove optional fields, or change required keys)