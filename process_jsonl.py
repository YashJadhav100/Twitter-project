import argparse, json, sys, os
import pandas as pd
from dateutil import parser as dtparser

def stream_json_lines(path, max_rows=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
            except Exception:
                # Skip malformed lines
                continue
            if max_rows and len(rows) >= max_rows:
                break
    return pd.DataFrame(rows)

def ensure_columns(df, required, optional):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: Missing required columns: {missing}. Available: {df.columns.tolist()}")
    # Fill optional columns that are missing
    for c in optional:
        if c not in df.columns:
            df[c] = None
    return df

def parse_dt_safe(x):
    if pd.isna(x):
        return pd.NaT
    try:
        return dtparser.parse(str(x))
    except Exception:
        return pd.NaT

def normalize(df):
    # Datetime
    if "created_at" in df.columns:
        df["created_at"] = df["created_at"].apply(parse_dt_safe)
    # Drop exact dupes by id if present
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])
    # Sort
    if "created_at" in df.columns:
        df = df.sort_values("created_at")
    return df

def main():
    ap = argparse.ArgumentParser(description="Clean & convert JSONL tweet-like data to Parquet + sample CSV")
    ap.add_argument("--input", required=True, help="Path to JSONL (one JSON object per line)")
    ap.add_argument("--schema", required=True, help="Path to schema.json")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument("--rows", type=int, default=None, help="Optional row cap for quick tests")
    ap.add_argument("--sample", type=int, default=1000, help="Rows to include in sample CSV (after cleaning)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.schema, "r", encoding="utf-8") as f:
        schema = json.load(f)
    required = schema.get("required", [])
    optional = schema.get("optional", [])

    print(f"Reading: {args.input}")
    df = stream_json_lines(args.input, max_rows=args.rows)
    print(f"Loaded shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    df = ensure_columns(df, required, optional)
    df = normalize(df)

    # Basic quality report
    report = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "null_created_at": int(df["created_at"].isna().sum()) if "created_at" in df.columns else None,
        "null_text": int(df["text"].isna().sum()) if "text" in df.columns else None,
        "lang_counts": df["lang"].value_counts().to_dict() if "lang" in df.columns else None,
    }
    print("Report:", json.dumps(report, indent=2))

    # Write outputs
    parquet_path = os.path.join(args.outdir, "data.parquet")
    sample_csv_path = os.path.join(args.outdir, "sample.csv")
    df.to_parquet(parquet_path, index=False)
    df.head(args.sample).to_csv(sample_csv_path, index=False)

    # Also write a tiny markdown summary
    summary_md = os.path.join(args.outdir, "SUMMARY.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Data Summary\n\n")
        for k, v in report.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\nFirst few rows (truncated):\n\n")
        f.write(df.head(5).to_markdown(index=False))

    print(f"Wrote: {parquet_path}")
    print(f"Wrote: {sample_csv_path}")
    print(f"Wrote: {summary_md}")

if __name__ == "__main__":
    main()