# python3 run_quality.py Qwen/Qwen2.5-0.5B-Instruct fp32
# python3 run_quality.py Qwen/Qwen2.5-3B-Instruct bf16
import sys, json, torch
from datasets import load_dataset
from evalkit import load_hf, make_loader, lm_bpb, sft_nll, eval_generation, bootstrap_ci

model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[2] if len(sys.argv) > 2 else "fp32"]
device = "cuda" if torch.cuda.is_available() else "cpu"

model, tok = load_hf(model_id, device, dtype=dtype)
texts = [t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(t.strip()) > 50][:200]
qa = [json.loads(l) for l in open("data/qa.jsonl") if l.strip()]
pairs = [{"prompt": x["question"], "response": x["answer"]} for x in qa]

out = {"meta": {"model": model_id, "dtype": str(dtype).split(".")[-1], "device": device,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "n_texts": len(texts), "max_length": 512, "n_qa": len(qa), "decoding": "greedy"}}
out["lm"] = lm_bpb(model, tok, make_loader(texts, tok, 4, 512), device)
out["sft"] = sft_nll(model, tok, pairs, device)
out["gen"] = eval_generation(model, tok, qa, device)
pt, lo, hi = bootstrap_ci([r["em"] for r in out["gen"]["records"]])
out["gen"]["em_ci95"] = [lo, hi]

tag = model_id.split("/")[-1].lower().replace(".", "").replace("-instruct", "") + "_" + out["meta"]["dtype"]
path = f"results/quality_{tag}.json"
json.dump(out, open(path, "w"), indent=2)
print(json.dumps({k: v for k, v in out["lm"].items()}, indent=2))
print("em:", out["gen"]["em"], "f1:", out["gen"]["f1"], "ci95:", out["gen"]["em_ci95"])
print("wrote", path)
