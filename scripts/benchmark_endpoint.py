#!/usr/bin/env python3
"""Concurrency sweep against an OpenAI-compatible /completions endpoint.

Reports latency percentiles (p50/p95/p99) and throughput in tokens/s and bytes/s.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

PROMPT = "Write a short paragraph about the weather today."


def one_request(endpoint, model, max_tokens):
    start = time.perf_counter()
    resp = requests.post(
        f"{endpoint}/completions",
        json={"model": model, "prompt": PROMPT, "max_tokens": max_tokens, "temperature": 0},
        timeout=120,
    )
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    body = resp.json()
    text = body["choices"][0]["text"]
    usage = body.get("usage", {})
    return {
        "latency_s": elapsed,
        "completion_tokens": usage.get("completion_tokens", 0),
        "completion_bytes": len(text.encode("utf-8")),
    }


def percentiles(values):
    s = sorted(values)

    def pct(p):
        return s[min(len(s) - 1, int(len(s) * p))]

    return {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99)}


def run_level(endpoint, model, concurrency, n_requests, max_tokens):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one_request, endpoint, model, max_tokens) for _ in range(n_requests)]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - start

    total_tokens = sum(r["completion_tokens"] for r in results)
    total_bytes = sum(r["completion_bytes"] for r in results)
    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "wall_time_s": wall,
        "latency_percentiles_s": percentiles([r["latency_s"] for r in results]),
        "throughput_tokens_per_s": total_tokens / wall if wall else 0,
        "throughput_bytes_per_s": total_bytes / wall if wall else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--concurrency-levels", nargs="+", type=int, default=[1, 5, 20])
    ap.add_argument("--requests-per-level", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", default="results/systems.json")
    args = ap.parse_args()

    one_request(args.endpoint, args.model, args.max_tokens)

    report = {"model": args.model, "max_tokens": args.max_tokens, "sweeps": []}
    for c in args.concurrency_levels:
        print(f"Running concurrency={c} ...")
        report["sweeps"].append(run_level(args.endpoint, args.model, c, args.requests_per_level, args.max_tokens))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
