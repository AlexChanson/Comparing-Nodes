import pandas as pd
from pathlib import Path
from typing import Tuple, Union

def average_time_columns_by_label_to_latex_pretty(
    csv_path: Union[str, Path],
    label_col: str = "label",
    float_precision: int = 3,
    output_csv: Union[str, Path, None] = None,
    output_tex: Union[str, Path, None] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Read a CSV of multiple runs, average all columns starting with 'time_' grouped by
    ['database', label] when 'database' exists, otherwise grouped by [label]. Keep
    non-time columns using the first value per group. Drop 'run' and any 'Unnamed:*'
    columns. Preserve the original column order from the CSV (minus the dropped ones).
    Sort rows by database then label (or just label). Return (DataFrame, LaTeX string).

    Parameters
    ----------
    csv_path : str | Path
        Input CSV path.
    label_col : str
        The label column name.
    float_precision : int
        Number of decimal places for numeric display in LaTeX.
    output_csv : optional
        If provided, write the averaged table to this CSV path.
    output_tex : optional
        If provided, write the LaTeX table to this path.

    Returns
    -------
    (df_out, latex_str) : (pd.DataFrame, str)
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found. Columns: {list(df.columns)}")

    # Drop 'run' and any Unnamed:* columns
    cols_to_drop = set()
    if "run" in df.columns:
        cols_to_drop.add("run")
    unnamed_mask = df.columns.str.match(r"^Unnamed", na=False)
    cols_to_drop.update(df.columns[unnamed_mask].tolist())

    df = df.drop(columns=list(cols_to_drop), errors="ignore")

    # Preserve original column order *after* drops
    original_cols = df.columns.tolist()

    # Choose grouping keys
    group_keys = ["database", label_col] if "database" in df.columns else [label_col]

    # Identify time_* columns (preserve their order as in the CSV)
    time_cols = [c for c in original_cols if c.startswith("time_")]

    # Identify non-time columns excluding group keys
    other_cols = [c for c in original_cols if c not in time_cols and c not in group_keys]

    # Group and aggregate
    g = df.groupby(group_keys, dropna=False)

    # Mean for time_* columns; 'first' for non-time columns
    avg_time = g[time_cols].mean(numeric_only=True) if len(time_cols) else pd.DataFrame(index=g.size().index)
    keep_other = g[other_cols].first() if len(other_cols) else pd.DataFrame(index=g.size().index)

    out = keep_other.join(avg_time)

    # Bring group keys back as columns
    out = out.reset_index()

    # Reorder columns exactly as in the CSV (after drops)
    final_cols = [c for c in original_cols if c in out.columns]
    out = out[final_cols]

    # Sort rows by database then label (or just label)
    sort_keys = [k for k in ["database", label_col] if k in out.columns]
    if sort_keys:
        out = out.sort_values(by=sort_keys, kind="stable")

    # ===== Prettify LaTeX =====
    # Build a column_format string: right-align numeric columns, left-align others
    num_cols = set(out.select_dtypes(include="number").columns.tolist())
    colfmt = "".join("r" if c in num_cols else "l" for c in out.columns)

    # Use Styler for better LaTeX with hrules and alignment
    styler = (
        out.style
        .hide(axis="index")
        .format(precision=float_precision)
    )
    latex_str = styler.to_latex(hrules=True, column_format=colfmt)

    # Optionally write files
    if output_csv:
        Path(output_csv).write_text(out.to_csv(index=False), encoding="utf-8")
    if output_tex:
        Path(output_tex).write_text(latex_str, encoding="utf-8")

    return out, latex_str


# ===== Example usage =====
if __name__ == "__main__":
    df_out, latex = average_time_columns_by_label_to_latex_pretty(
        csv_path="reports/results_18-10-25:15:38:04.csv",
        label_col="label",
        float_precision=2,
        output_csv="averaged_time_by_label.csv",
        output_tex="averaged_time_by_label.tex",
    )
    print(df_out.head())
    print("\n===== LaTeX Preview =====\n")
    print(latex)
