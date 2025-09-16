import json, argparse, sys

def normalize_keys(obj):
    # replace dots in keys: "author.id" -> "author_id"
    fixed = {}
    for k, v in obj.items():
        k2 = k.replace(".", "_")
        fixed[k2] = v
    # rename fields to our expected schema
    if "CreatedAt" in fixed and "created_at" not in fixed:
        fixed["created_at"] = fixed.pop("CreatedAt")
    # normalize count field names
    if "likeCount" in fixed and "like_count" not in fixed:
        fixed["like_count"] = fixed.pop("likeCount")
    if "retweetCount" in fixed and "retweet_count" not in fixed:
        fixed["retweet_count"] = fixed.pop("retweetCount")
    if "replyCount" in fixed and "reply_count" not in fixed:
        fixed["reply_count"] = fixed.pop("replyCount")
    if "quoteCount" in fixed and "quote_count" not in fixed:
        fixed["quote_count"] = fixed.pop("quoteCount")
    if "viewCount" in fixed and "view_count" not in fixed:
        fixed["view_count"] = fixed.pop("viewCount")
    return fixed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to JSON array file")
    ap.add_argument("--output", required=True, help="Path to write JSONL")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)  # expects a top-level list/array

    if not isinstance(data, list):
        print("ERROR: Input JSON is not a top-level array/list.", file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as out:
        for obj in data:
            if not isinstance(obj, dict):
                continue
            obj = normalize_keys(obj)
            # keep only rows that have essential fields after renaming
            if "id" in obj and "text" in obj:
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote JSONL to: {args.output}")

if __name__ == "__main__":
    main()
