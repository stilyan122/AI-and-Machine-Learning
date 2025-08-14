"""
Deterministic data-prep pipeline:
- loads raw/preprocessed CSV
- applies our reusable helpers/fe functions
- writes a stable, float-only feature table
"""

from pathlib import Path
import argparse
import pandas as pd

from src.seed import set_seed
from src.helpers import drop_and_dedup, build_column_order, scale_continuous
from src.feat import (
    fe_add_composites, fe_add_interactions, fe_one_hot_codes,
    fe_variance_filter, fe_drop_high_corr, fe_enforce_float_and_order,
    fe_build_feature_table,
)

TARGET = "Diagnosis"
CODE_COLS = ["Gender", "Ethnicity", "EducationLevel"]

def run(input_csv: str, output_csv: str, seed: int = 42) -> None:
    set_seed(seed)

    # load
    df = pd.read_csv(input_csv)

    # Minimal cleanup (dedup + drop obvious id/doctor columns if present)
    df, _ = drop_and_dedup(df, drop_cols=["PatientID","DoctorInCharge"])

    # Build features using your unified function (keeps order, floats, pruning)
    final, info = fe_build_feature_table(
        df,
        target=TARGET,
        code_cols=[c for c in CODE_COLS if c in df.columns],
        corr_threshold=0.98,
        drop_first=False,
    )

    # Deterministic save: stable row index & column order are already ensured
    final = final.reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_csv, index=False, float_format="%.6f")

    # 5) Llog a tiny manifest for audit
    manifest = {
        "input": str(Path(input_csv).resolve()),
        "output": str(Path(output_csv).resolve()),
        "n_rows": int(final.shape[0]),
        "n_cols": int(final.shape[1]),
        "seed": int(seed),
        "added_by_composites": info.get("added_by_composites", []),
        "added_by_interactions": info.get("added_by_interactions", []),
        "one_hot_new_cols": info.get("one_hot_new_cols", []),
        "dropped_zero_var": info.get("dropped_zero_var", []),
        "dropped_high_corr": info.get("dropped_high_corr", []),
    }
    
    pd.Series(manifest, dtype="object").to_json(
        Path(output_csv).with_suffix(".manifest.json"), indent=2
    )

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="outp", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(args.inp, args.outp, seed=args.seed)

if __name__ == "__main__":
    cli()