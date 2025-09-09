#!/usr/bin/env python3
"""
Process a CSV so that all columns except the first are filtered and percentile-scaled.

Rules applied to each column (except the first):
  (i)  compute null_ratio = (#null / #rows); drop if null_ratio > --null-thresh
  (ii) compute distinct_ratio = (#distinct_non_null / #non_null);
       drop if distinct_ratio < --distinct-low or distinct_ratio > --distinct-high
  (iii) otherwise normalize using percentile scaling (empirical CDF in [0,1])

Notes:
- Distinct ratio is computed over non-null values. If a column is all-null, it's dropped.
- By default, only numeric columns are considered for scaling; non-numeric columns are excluded.
  Use --allow-non-numeric to allow percentile scaling on orderable non-numeric types (e.g., strings).
- First column is preserved as-is.

Usage:
  python process_df.py --input data.csv --output processed.csv --report report.csv \
      --null-thresh 0.3 --distinct-low 0.02 --distinct-high 0.98
"""

import argparse
import sys
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np


def percentile_scale(s: pd.Series) -> pd.Series:
    """
    Percentile scaling: map non-null values to [0,1] based on their rank.
    - If all non-null values are equal, map them to 0.5.
    - Nulls remain null.
    Exact mapping:
      Let r = rank(method='average') in [1, n_non_null]; return (r - 1) / (n_non_null - 1)
      so min -> 0 and max -> 1 when n_non_null > 1.
    """
    s_out = s.copy()
    non_null = s.dropna()
    n = len(non_null)
    if n == 0:
        return s  # all nulls; leave as-is (will likely be dropped by rules anyway)
    if non_null.nunique(dropna=True) == 1:
        # Constant column: map all non-nulls to 0.5
        s_out.loc[non_null.index] = 0.5
        return s_out

    ranks = non_null.rank(method="average")  # 1..n
    scaled = (ranks - 1) / (n - 1)           # 0..1
    s_out.loc[non_null.index] = scaled
    return s_out


def process_dataframe(
    df: pd.DataFrame,
    null_thresh: float,
    distinct_low: float,
    distinct_high: float,
    allow_non_numeric: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process df as specified. Returns (processed_df, report_df).

    report_df columns:
      - column
      - action: 'kept_scaled' | 'dropped'
      - reason: for dropped columns (e.g., 'null_ratio>thr', 'distinct_ratio<low', 'distinct_ratio>high', 'non_numeric')
      - null_ratio
      - distinct_ratio
      - dtype_before
    """
    if df.shape[1] < 2:
        raise ValueError("DataFrame must have at least two columns (first preserved, others processed).")

    first_col = df.columns[0]
    keep_df = df[[first_col]].copy()

    report_rows: List[Dict[str, Any]] = []

    total_rows = len(df)

    for col in df.columns[1:]:
        s_orig = df[col]
        dtype_before = str(s_orig.dtype)

        # Compute ratios
        null_ratio = s_orig.isna().mean() if total_rows > 0 else np.nan

        non_null = s_orig.dropna()
        non_null_count = len(non_null)
        if non_null_count == 0:
            distinct_ratio = np.nan
        else:
            distinct_ratio = non_null.nunique(dropna=True) / non_null_count

        # Rule (i): null ratio threshold
        if pd.notna(null_ratio) and null_ratio > null_thresh:
            report_rows.append({
                "column": col,
                "action": "dropped",
                "reason": "null_ratio>thr",
                "null_ratio": null_ratio,
                "distinct_ratio": distinct_ratio,
                "dtype_before": dtype_before,
            })
            continue

        # Type handling
        s_for_scale = s_orig
        is_numeric = pd.api.types.is_numeric_dtype(s_for_scale)

        if not is_numeric and not allow_non_numeric:
            # Try coercion to numeric; if coercion creates many NaNs, we still enforce numeric-only unless allowed
            coerced = pd.to_numeric(s_for_scale, errors="coerce")
            if pd.api.types.is_numeric_dtype(coerced):
                s_for_scale = coerced
                is_numeric = True
            else:
                report_rows.append({
                    "column": col,
                    "action": "dropped",
                    "reason": "non_numeric",
                    "null_ratio": null_ratio,
                    "distinct_ratio": distinct_ratio,
                    "dtype_before": dtype_before,
                })
                continue

        # Rule (ii): distinct ratio thresholds (over non-null entries)
        # If non_null_count == 0 -> drop (degenerate)
        if non_null_count == 0:
            report_rows.append({
                "column": col,
                "action": "dropped",
                "reason": "all_null",
                "null_ratio": null_ratio,
                "distinct_ratio": distinct_ratio,
                "dtype_before": dtype_before,
            })
            continue

        if pd.notna(distinct_ratio) and distinct_ratio < distinct_low:
            report_rows.append({
                "column": col,
                "action": "dropped",
                "reason": "distinct_ratio<low",
                "null_ratio": null_ratio,
                "distinct_ratio": distinct_ratio,
                "dtype_before": dtype_before,
            })
            continue

        if pd.notna(distinct_ratio) and distinct_ratio > distinct_high:
            report_rows.append({
                "column": col,
                "action": "dropped",
                "reason": "distinct_ratio>high",
                "null_ratio": null_ratio,
                "distinct_ratio": distinct_ratio,
                "dtype_before": dtype_before,
            })
            continue

        # Rule (iii): percentile scaling
        s_scaled = percentile_scale(s_for_scale)
        keep_df[col] = s_scaled

        report_rows.append({
            "column": col,
            "action": "kept_scaled",
            "reason": "",
            "null_ratio": null_ratio,
            "distinct_ratio": distinct_ratio,
            "dtype_before": dtype_before,
        })

    report_df = pd.DataFrame(report_rows, columns=[
        "column", "action", "reason", "null_ratio", "distinct_ratio", "dtype_before"
    ])

    return keep_df, report_df

def export(processed_df,report_df,output,report):
        try:
            processed_df.to_csv(output, index=False)
            report_df.to_csv(report, index=False)
        except Exception as e:
            print(f"Failed to write outputs: {e}", file=sys.stderr)
            sys.exit(1)

            # Brief console summary
        #kept = (report_df["action"] == "kept_scaled").sum()
        #dropped = (report_df["action"] == "dropped").sum()
        #print(f"Processed. Kept+scaled: {kept}, Dropped: {dropped}.")
        print(f"Saved processed CSV to: {output}")
        print(f"Saved report CSV to:    {report}")



def main():
    p = argparse.ArgumentParser(description="Filter and percentile-scale columns (except the first).")
    p.add_argument("--input", required=True, help="Input CSV file")
    p.add_argument("--output", required=True, help="Output CSV file (processed)")
    p.add_argument("--report", required=True, help="Report CSV file (kept/dropped with reasons)")
    p.add_argument("--null-thresh", type=float, default=0.3, help="Drop if null ratio > this (default: 0.30)")
    p.add_argument("--distinct-low", type=float, default=0.02, help="Drop if distinct ratio < this (default: 0.02)")
    p.add_argument("--distinct-high", type=float, default=0.98, help="Drop if distinct ratio > this (default: 0.98)")
    p.add_argument("--allow-non-numeric", action="store_true",
                   help="Allow percentile scaling for non-numeric columns (e.g., strings)")

    args = p.parse_args()

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Failed to read input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    processed_df, report_df = process_dataframe(
        df,
        null_thresh=args.null_thresh,
        distinct_low=args.distinct_low,
        distinct_high=args.distinct_high,
        allow_non_numeric=args.allow_non_numeric
    )

    try:
        processed_df.to_csv(args.output, index=False)
        report_df.to_csv(args.report, index=False)
    except Exception as e:
        print(f"Failed to write outputs: {e}", file=sys.stderr)
        sys.exit(1)

    # Brief console summary
    kept = (report_df["action"] == "kept_scaled").sum()
    dropped = (report_df["action"] == "dropped").sum()
    print(f"Processed. Kept+scaled: {kept}, Dropped: {dropped}.")
    print(f"Saved processed CSV to: {args.output}")
    print(f"Saved report CSV to:    {args.report}")


if __name__ == "__main__":
    main()
