import pandas as pd

# Load both files
main_df = pd.read_parquet('data/processed/race_scores.parquet')
conn_df = pd.read_parquet('data/processed/race_scores_connections_v2.parquet')

print("Column comparison:")
print(f"\nrace_scores.parquet columns: {len(main_df.columns)}")
print(f"connections_v2.parquet columns: {len(conn_df.columns)}")

# Show columns in connections_v2 that aren't in race_scores
conn_only = set(conn_df.columns) - set(main_df.columns)
print(f"\nColumns ONLY in connections_v2 ({len(conn_only)}):")
for col in sorted(conn_only):
    print(f"  - {col}")

# Show columns in race_scores that aren't in connections_v2
main_only = set(main_df.columns) - set(conn_df.columns)
print(f"\nColumns ONLY in race_scores ({len(main_only)}):")
for col in sorted(main_only):
    print(f"  - {col}")

print(f"\nCommon columns: {len(set(main_df.columns) & set(conn_df.columns))}")
