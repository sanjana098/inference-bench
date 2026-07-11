import argparse
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/ttft_all.png")
ap.add_argument("--log", action="store_true", help="log-log axes")
args = ap.parse_args()

dtype_order = {"fp32": 0, "bf16": 1}
runs = []
for path in glob.glob("results/ttft_*.json"):
    d = json.load(open(path))
    if "rows" in d:
        runs.append(d)

if not runs:
    raise SystemExit("no ttft_*.json files with a 'rows' key found in results/")

runs.sort(key=lambda d: (d["model"], dtype_order.get(d["dtype"], 9)))

for d in runs:
    x = [r["prompt_tokens"] for r in d["rows"]]
    y = [1000 * r["ttft_p50_s"] for r in d["rows"]]
    name = d["model"].split("/")[-1].replace("-Instruct", "")
    plt.plot(x, y, marker="o", label=f"{name} {d['dtype']}")
    plt.annotate(f"{y[-1]:.0f} ms", (x[-1], y[-1]), textcoords="offset points",
                 xytext=(6, 0), va="center", fontsize=8)

if args.log:
    plt.xscale("log")
    plt.yscale("log")

plt.margins(x=0.08)
plt.xlabel("Prompt tokens")
plt.ylabel("TTFT p50 (ms)")
plt.title(f"TTFT vs prompt length: Qwen2.5 on {runs[0]['gpu']}")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print("wrote", args.out, "from", len(runs), "runs")
