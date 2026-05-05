import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_feature_ranges_from_csv(csv_path, feature_names):
    df = pd.read_csv(csv_path)
    feature_ranges = {}
    
    for feat in feature_names:
        if feat not in df.columns:
            raise ValueError(f"Column '{feat}' not found in the CSV: {csv_path}")
        
        col_data = df[feat].dropna()
        if col_data.empty:
            feature_ranges[feat] = (0.0, 1.0)
        else:
            # Force the minimum to 0 to avoid the extreme zoom effect
            c_min = float(col_data.min())
            c_max = float(col_data.max())
            
            if c_min > 0:
                c_min = 0.0  # Center of the radar is always 0
                
            if c_max <= c_min:
                c_max = c_min + 1.0
                
            feature_ranges[feat] = (c_min, c_max)
            
    return feature_ranges


def plot_pair_radar(
    feature_names, a_values, b_values, *,
    feature_ranges, a_label="A", b_label="B",
    title=None, outpath=None, color_a="tab:blue",
    color_b="tab:orange", clip=True
):
    feats = list(feature_names)
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)

    if len(feats) == 0:
        return

    # Handle NaNs and infinities
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

    # Extract ranges
    ranges = []
    for f in feats:
        mn, mx = feature_ranges[f]
        ranges.append((float(mn), float(mx)))

    # Scale values to [0, 1]
    a_scaled = np.empty_like(a, dtype=float)
    b_scaled = np.empty_like(b, dtype=float)
    for i, (mn, mx) in enumerate(ranges):
        a_scaled[i] = (a[i] - mn) / (mx - mn)
        b_scaled[i] = (b[i] - mn) / (mx - mn)

    if clip:
        a_scaled = np.clip(a_scaled, 0.0, 1.0)
        b_scaled = np.clip(b_scaled, 0.0, 1.0)

    # Close the polygon loop
    N = len(feats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    a_plot = np.concatenate([a_scaled, a_scaled[:1]])
    b_plot = np.concatenate([b_scaled, b_scaled[:1]])

    # Initialize plot
    fig = plt.figure(figsize=(9, 6.5))
    ax = plt.subplot(111, polar=True)

    # Plot A
    ax.plot(angles, a_plot, linewidth=2, label=a_label, color=color_a)
    ax.fill(angles, a_plot, alpha=0.10, color=color_a)

    # Plot B
    ax.plot(angles, b_plot, linewidth=2, label=b_label, color=color_b)
    ax.fill(angles, b_plot, alpha=0.10, color=color_b)

    # Set axes and labels
    ax.set_xticks(angles[:-1])
    xticklabels = [f"{f}\n[{mn:g}, {mx:g}]" for f, (mn, mx) in zip(feats, ranges)]
    ax.set_xticklabels(xticklabels, fontsize=8)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)

    if title:
        ax.set_title(title, pad=20, fontsize=11)
    else:
        ax.set_title("Pair radar comparison", pad=20, fontsize=11)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    ax.grid(True)
    fig.tight_layout()

    # Save or show
    if outpath:
        outdir = os.path.dirname(outpath)
        if outdir: 
            os.makedirs(outdir, exist_ok=True)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        print(f"Saved radar chart to: {outpath}")
        plt.close(fig)
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Generate radar charts from CSV.")
    parser.add_argument("--csv", required=True, help="Path to the reference CSV file.")
    parser.add_argument("--title", default="Radar Comparison", help="Chart title.")
    parser.add_argument("--out", default=None, help="Output image path.")
    
    args = parser.parse_args()
    df = pd.read_csv(args.csv)

    if len(df) < 2:
        print("Error: CSV must contain at least 2 rows to compare.")
        return
    
    # Auto-extract names from the first column
    label_a = str(df.iloc[0, 0])
    label_b = str(df.iloc[1, 0])
    features = list(df.columns[1:])
    
    # Extract data values
    data_matrix = df.iloc[:, 1:].values.astype(float)
    val_a = data_matrix[0].tolist()
    val_b = data_matrix[1].tolist()

    # Calculate ranges (now automatically forcing 0 as minimum)
    ranges = get_feature_ranges_from_csv(args.csv, features)
    
    plot_pair_radar(
        feature_names=features,
        a_values=val_a,
        b_values=val_b,
        feature_ranges=ranges,
        a_label=label_a,
        b_label=label_b,
        title=args.title,
        outpath=args.out
    )

if __name__ == "__main__":
    main()