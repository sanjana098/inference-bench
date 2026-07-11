# python3 -i session.py  — loads model, tokenizer, data; run inside tmux
import json, math, torch, torch.nn.functional as F
from datasets import load_dataset
from evalkit import *

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
model, tok = load_hf(MODEL, device)
texts = [t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(t.strip()) > 50][:200]
loader = make_loader(texts, tok, 4, 512)
qa = [json.loads(l) for l in open("data/qa.jsonl") if l.strip()]
pairs = [{"prompt": x["question"], "response": x["answer"]} for x in qa]
print(f"ready: {MODEL} on {device}, {len(texts)} texts, {len(qa)} qa")
