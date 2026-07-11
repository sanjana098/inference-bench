import argparse
import json
import time

import torch

from evalkit import load_hf

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"],
                help="fp32 is the reference precision; pass bf16 for the serving precision")
args = ap.parse_args()

dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
device = "cuda"

model, tok = load_hf(args.model, device, dtype=dtype)
gpu = torch.cuda.get_device_name(0)
short = args.model.split("/")[-1]
name = short.replace("-Instruct", "")
tag = name.lower().replace(".", "") + "_" + args.dtype

base = "The history of the Roman Empire spans centuries of expansion, conflict, and reform. "
targets = [100, 500, 1000, 2000, 4000]
WARMUP, RUNS = 2, 5

rows = []
for n in targets:
    text = base * (n // 15 + 1)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=n).to(device)
    L = ids["input_ids"].shape[1]
    times = []
    with torch.inference_mode():
        for i in range(WARMUP + RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.generate(**ids, max_new_tokens=1, do_sample=False, pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= WARMUP:
                times.append(t1 - t0)
    times.sort()
    rows.append({"prompt_tokens": L, "ttft_p50_s": times[len(times) // 2],
                 "ttft_min_s": times[0], "ttft_max_s": times[-1]})
    print(rows[-1])

json.dump({"model": args.model, "dtype": args.dtype, "gpu": gpu,
           "warmup_runs": WARMUP, "measured_runs": RUNS, "rows": rows},
          open(f"results/ttft_{tag}.json", "w"), indent=2)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot([r["prompt_tokens"] for r in rows], [1000 * r["ttft_p50_s"] for r in rows], marker="o")
    plt.xlabel("Prompt tokens")
    plt.ylabel("TTFT p50 (ms)")
    plt.title(f"TTFT vs prompt length: {name} {args.dtype}, {gpu}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/ttft_{tag}.png", dpi=150)
    print(f"wrote results/ttft_{tag}.png")
except Exception as e:
    print("chart skipped:", e)