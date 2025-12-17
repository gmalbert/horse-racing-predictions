import os
import glob
import pandas as pd


def read_csv_flexible(path):
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    root = os.path.abspath(root)
    raw_dir = os.path.join(root, "data", "raw")
    pattern = os.path.join(raw_dir, "*.csv")
    files = sorted(glob.glob(pattern))

    out_path = os.path.join(raw_dir, "all_gb_races.csv")

    # exclude output if it already exists in the list
    files = [f for f in files if os.path.basename(f) != os.path.basename(out_path)]

    if not files:
        print("No CSV files found in", raw_dir)
        return

    dfs = []
    all_columns = []

    for f in files:
        print("Reading", os.path.relpath(f, root))
        df = read_csv_flexible(f)
        # normalize column names
        df.columns = df.columns.str.strip()
        dfs.append(df)
        all_columns.extend(list(df.columns))

    # build union of columns preserving order (first seen)
    seen = []
    for c in all_columns:
        if c not in seen:
            seen.append(c)

    union_cols = seen

    aligned = []
    for df, f in zip(dfs, files):
        # reindex to union; missing columns become NaN
        df = df.reindex(columns=union_cols)
        # add a source column with filename
        df["_source_file"] = os.path.basename(f)
        aligned.append(df)

    print("Concatenating %d files -> %s" % (len(aligned), os.path.relpath(out_path, root)))
    result = pd.concat(aligned, ignore_index=True, sort=False)

    # write output
    result.to_csv(out_path, index=False)
    print("Wrote", out_path, "rows=", len(result))


if __name__ == "__main__":
    main()
