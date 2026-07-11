import json, time, torch
from evalkit import load_hf

device = "cuda"
model, tok = load_hf("Qwen/Qwen2.5-0.5B-Instruct", device)
base = "The history of the Roman Empire spans centuries of expansion, conflict, and reform. "
targets = [100, 500, 1000, 2000, 4000]
rows = []
for n in targets:
    text = base * (n // 15 + 1)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=n).to(device)
    L = ids["input_ids"].shape[1]
    times = []
    with torch.inference_mode():
        for i in range(5):                                   # 2 warm-up + 3 measured
            torch.cuda.synchronize(); t0 = time.perf_counter()
            model.generate(**ids, max_new_tokens=1, do_sample=False, pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            if i >= 2: times.append(t1 - t0)
    times.sort(); rows.append({"prompt_tokens": L, "ttft_p50_s": times[1]})
    print(rows[-1])
json.dump({"model": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "fp32", "gpu": torch.cuda.get_device_name(0),
           "prompt_tokens": [r["prompt_tokens"] for r in rows], "ttft_p50_s": [r["ttft_p50_s"] for r in rows]},
          open("results/ttft_vs_prompt_len.json", "w"), indent=2)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.plot([r["prompt_tokens"] for r in rows], [1000 * r["ttft_p50_s"] for r in rows], marker="o")
    plt.xlabel("prompt tokens"); plt.ylabel("TTFT p50 (ms)")
    plt.title("TTFT vs prompt length: Qwen2.5-0.5B fp32, NVIDIA A30")
    plt.tight_layout()
    plt.grid(alpha=.3); plt.savefig("results/ttft_vs_prompt.png", dpi=150, bbox_inches="tight"); print("wrote results/ttft_vs_prompt.png")
except Exception as e:
    print("chart skipped:", e)
