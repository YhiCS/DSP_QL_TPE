"""
aggregate.py

Concatenates every per-(participant × env) paramFit pickle in dump/paramFit/
into one combined DataFrame and writes it to dump/paramFit/paramFit_ALL.pkl.

Run after a SLURM array completes, e.g.:
    sbatch --dependency=afterok:<ARRAY_JOB_ID> aggregate.sh
or interactively:
    python aggregate.py
"""

from pathlib import Path
import pandas as pd

_HERE     = Path(__file__).resolve().parent
DUMP_DIR  = _HERE / "dump" / "paramFit"
OUT_PATH  = DUMP_DIR / "paramFit_ALL.pkl"

STAGE_ORDER = ["group", "S1_pair", "S2_pair"]


def main():
    pkls = sorted(p for p in DUMP_DIR.glob("paramFit_*.pkl") if p != OUT_PATH)
    if not pkls:
        print(f"No paramFit_*.pkl files found in {DUMP_DIR}")
        return

    dfs = [pd.read_pickle(p) for p in pkls]
    combined = pd.concat(dfs, ignore_index=True)

    if "stage" in combined.columns:
        combined["stage"] = pd.Categorical(
            combined["stage"], categories=STAGE_ORDER, ordered=True
        )

    combined.to_pickle(OUT_PATH)
    n_subj = combined["subjID"].nunique() if "subjID" in combined.columns else "?"
    n_env  = combined["env"].nunique()    if "env"    in combined.columns else "?"
    print(
        f"Combined {len(pkls)} files → {OUT_PATH} "
        f"({len(combined)} rows, {n_subj} subjects, {n_env} envs)"
    )


if __name__ == "__main__":
    main()
