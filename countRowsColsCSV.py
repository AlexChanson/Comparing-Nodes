import csv
import os


def summarize_csv_directory(directory_path) -> None:
    """Prints the number of rows (excluding header) and columns for each CSV file in the given directory."""
    # List all files ending with .csv in the directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the directory.")
        return

    print(f"{'File':<40} {'Rows':>10} {'Columns':>10}")
    print("-" * 62)

    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        try:
            with open(file_path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                n_cols = len(header) if header else 0
                n_rows = sum(1 for _ in reader)
            print(f"{csv_file:<40} {n_rows:>10} {n_cols:>10}")
        except Exception as e:
            print(f"{csv_file:<40} {'ERROR':>10} {e!s:>10}")


# ===== Example usage =====
if __name__ == "__main__":
    summarize_csv_directory("sample_data/indicators_171025")
