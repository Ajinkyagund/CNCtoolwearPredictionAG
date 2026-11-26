"""
Generates demo tool history CSV files from held-out test tools for the Tool Analysis tab.
Derived from the NASA / UC Berkeley milling dataset.
"""

from pathlib import Path
import json
import pandas as pd


def get_paths():
    candidates = [
        (Path("models"), Path("outputs")),
        (Path(__file__).resolve().parent / "models", Path(__file__).resolve().parent / "outputs"),
        (Path(__file__).resolve().parent.parent / "models", Path(__file__).resolve().parent.parent / "outputs"),
    ]
    for m, o in candidates:
        if (m / "meta.json").exists() and (o / "processed_features.csv").exists():
            return m, o
    return Path("models"), Path("outputs")


def make_demo_data():
    models_dir, outputs_dir = get_paths()
    meta_path = models_dir / "meta.json"
    data_path = outputs_dir / "processed_features.csv"

    if not meta_path.exists() or not data_path.exists():
        print(f"Error: Required files not found at {meta_path} or {data_path}")
        return

    meta = json.loads(meta_path.read_text())
    feats = meta["features"]
    df = pd.read_csv(data_path)

    # Filter to held-out test tools only
    test_df = df[df["split"] == "test"].copy()
    test_cases = sorted(test_df["case"].unique())

    # Create demo_data folders in both current script directory and root workspace
    dest_dirs = [
        Path(__file__).resolve().parent / "demo_data",
        Path(__file__).resolve().parent.parent / "demo_data",
    ]

    for d in dest_dirs:
        d.mkdir(parents=True, exist_ok=True)

    keep_cols = ["case", "run"]
    if "VB" in test_df.columns:
        keep_cols.append("VB")
    if "RUL" in test_df.columns:
        keep_cols.append("RUL")
    keep_cols.extend([f for f in feats if f not in keep_cols])

    for case_id in test_cases:
        tool_runs = test_df[test_df["case"] == case_id].sort_values("run")[keep_cols]
        fname = f"tool_{int(case_id):02d}.csv"
        for d in dest_dirs:
            tool_runs.to_csv(d / fname, index=False)
        print(f"Wrote {fname} ({len(tool_runs)} runs)")

    # Write a partial tool example (first 5 runs of test tool 03)
    partial_case = test_cases[0]
    partial_runs = test_df[test_df["case"] == partial_case].sort_values("run")[keep_cols].head(5)
    for d in dest_dirs:
        partial_runs.to_csv(d / "tool_XX_partial.csv", index=False)
        readme = d / "README.md"
        readme.write_text("Derived from the NASA milling dataset (held-out test tools). Not real shop data.\n")
    print(f"Wrote tool_XX_partial.csv ({len(partial_runs)} runs) and demo_data/README.md")


if __name__ == "__main__":
    make_demo_data()
