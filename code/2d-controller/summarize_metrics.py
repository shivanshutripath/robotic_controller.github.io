import pandas as pd
import sys

csv_path = sys.argv[1] if len(sys.argv) > 1 else "metrics_all.csv"
df = pd.read_csv(csv_path)

# success iteration per run (first iter where failed==0 and errors==0)
df["is_success"] = (df["failed"].fillna(1) == 0) & (df["errors"].fillna(1) == 0)

first_success = (
    df[df["is_success"]]
    .sort_values(["run_id", "iter"])
    .groupby(["model", "run_id"], as_index=False)
    .first()[["model","run_id","iter","t_total_s"]]
    .rename(columns={"iter":"success_iter","t_total_s":"t_total_at_success"})
)

# total runtime per run (sum t_total_s across iters)
totals = (
    df.groupby(["model","run_id"], as_index=False)["t_total_s"]
    .sum()
    .rename(columns={"t_total_s":"t_total_sum"})
)

out = totals.merge(first_success, on=["model","run_id"], how="left")
print(out.sort_values(["model","run_id"]).to_string(index=False))

print("\nBy model:")
print(out.groupby("model")[["t_total_sum","success_iter"]].agg(["mean","std","min","max","count"]))
