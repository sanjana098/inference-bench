#!/usr/bin/env bash
# machine_report.sh — collects everything needed to judge whether a GPU machine is
# viable for LLM inference benchmarking. Works on Ubuntu and CentOS/Rocky/Alma.
#
# Usage:
#   bash machine_report.sh
#   bash machine_report.sh --docker-test   # also runs the containerized-GPU check
#
# Output is printed to the screen AND saved to machine_report.txt in the current directory.

OUT="machine_report.txt"
DOCKER_TEST=0
[[ "$1" == "--docker-test" ]] && DOCKER_TEST=1

section() { echo; echo "=== $1 ==="; }
have()    { command -v "$1" >/dev/null 2>&1; }

{
  echo "machine_report generated: $(date)"
  echo "hostname: $(hostname)"

  section "OS"
  [[ -f /etc/os-release ]] && head -4 /etc/os-release || echo "no /etc/os-release"
  echo "kernel: $(uname -r)"
  if have ldd; then echo "glibc: $(ldd --version 2>/dev/null | head -1)"; fi

  section "GPU"
  if have nvidia-smi; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap --format=csv
  elif have rocm-smi; then
    echo "(AMD GPU detected)"
    rocm-smi --showproductname --showmeminfo vram
  else
    echo "no nvidia-smi or rocm-smi found — no GPU tooling installed, or not on PATH"
  fi

  section "GPU TOPOLOGY (interconnect, only meaningful with 2+ GPUs)"
  if have nvidia-smi; then nvidia-smi topo -m 2>/dev/null || echo "topo not available"; else echo "n/a"; fi

  section "CUDA TOOLKIT (optional — vLLM bundles its own runtime, driver version is what matters)"
  if have nvcc; then nvcc --version | tail -1; else echo "nvcc not installed (this is fine)"; fi

  section "CPU"
  lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)"

  section "RAM"
  free -h

  section "DISK"
  echo "-- filesystems (root and home):"
  df -h / "$HOME" 2>/dev/null
  echo "-- block devices (ROTA=0 means SSD, ROTA=1 means spinning disk):"
  lsblk -d -o NAME,SIZE,ROTA,TYPE 2>/dev/null

  section "DOCKER"
  if have docker; then
    docker --version 2>&1
    echo "-- runtimes:"
    docker info 2>/dev/null | grep -i -A3 "runtimes" || echo "could not read docker info (try: sudo docker info, or add yourself to the docker group)"
  else
    echo "docker not installed"
  fi

  section "PYTHON"
  if have python3; then python3 --version; else echo "python3 not found"; fi
  if have pip3; then pip3 --version 2>&1; else echo "pip3 not found"; fi

  section "GPU PROCESSES CURRENTLY RUNNING (is anyone else using it?)"
  if have nvidia-smi; then
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null
  else
    echo "n/a"
  fi

  if [[ $DOCKER_TEST -eq 1 ]]; then
    section "CONTAINERIZED GPU TEST (definitive check that docker can see the GPU)"
    if have docker; then
      docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi 2>&1 \
        || echo "FAILED — containerized GPU access not working. Usually means nvidia-container-toolkit isn't installed or configured."
    else
      echo "docker not installed, skipping"
    fi
  else
    echo
    echo "(Skipped containerized GPU test. Re-run with --docker-test to include it; it pulls a ~100MB image.)"
  fi

} 2>&1 | tee "$OUT"

echo
echo "Saved to: $(pwd)/$OUTpts_machine_report_sh

cat > scripts/benchmark_endpoint.py << 'EOF_scripts_benchmark_endpoint_py'
#!/usr/bin/env python3
"""
Systems evals -- how fast and cheap is it to serve, under realistic concurrency.

Reports latency as percentiles (p50/p95/p99), not just an average -- tail latency is
what users actually feel, and averages hide it. Sweeps concurrency (1, 5, 20 by default)
instead of testing at one fixed load, since throughput and latency trade off against each
other as concurrency rises, and a single data point can't show you that curve.

Reports throughput two ways: tokens/sec (the usual metric) and bytes/sec (the metric
that's actually comparable if you're serving a byte-level model, where "a token" isn't
a stable unit).

Usage:
    python benchmark_systems.py --endpoint http://localhost:8000/v1
"""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

PROMPT = "Write a short paragraph about the weather today."


def one_request(endpoint: str, model: str, max_tokens: int) -> dict:
    start = time.perf_counter()
    resp = requests.post(
        f"{endpoint}/completions",
        json={"model": model, "prompt": PROMPT, "max_tokens": max_tokens, "temperature": 0.7},
        timeout=120,
    )
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    text = choice["text"]
    usage = resp.json().get("usage", {})
    return {
        "latency_s": elapsed,
        "completion_tokens": usage.get("completion_tokens", 0),
        "completion_bytes": len(text.encode("utf-8")),
    }


def percentiles(values: list[float]) -> dict:
    s = sorted(values)
    def pct(p):
        idx = min(len(s) - 1, int(len(s) * p))
        return s[idx]
    return {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99)}


def run_concurrency_level(endpoint: str, model: str, concurrency: int, n_requests: int, max_tokens: int) -> dict:
    results = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one_request, endpoint, model, max_tokens) for _ in range(n_requests)]
        for f in futures:
            results.append(f.result())
    wall_time = time.perf_counter() - start

    latencies = [r["latency_s"] for r in results]
    total_tokens = sum(r["completion_tokens"] for r in results)
    total_bytes = sum(r["completion_bytes"] for r in results)

    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "wall_time_s": wall_time,
        "latency_percentiles_s": percentiles(latencies),
        "throughput_tokens_per_s": total_tokens / wall_time if wall_time else 0,
        "throughput_bytes_per_s": total_bytes / wall_time if wall_time else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--concurrency-levels", nargs="+", type=int, default=[1, 5, 20])
    ap.add_argument("--requests-per-level", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", default="results/systems.json")
    args = ap.parse_args()

    report = {"model": args.model, "sweeps": []}
    for c in args.concurrency_levels:
        print(f"Running concurrency={c} ...")
        report["sweeps"].append(
            run_concurrency_level(args.endpoint, args.model, c, args.requests_per_level, args.max_tokens)
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
EOF_scripts_benchmark_endpoint_py

cat > scripts/regression_gate.py << 'EOF_scripts_regression_gate_py'
#!/usr/bin/env python3
"""
The CI/CD gate -- diffs today's results against a stored baseline and fails (non-zero
exit code) if anything regressed past a threshold. This is the literal mechanism behind
"automate the path from research checkpoints to validated deployments": nothing promotes
to production unless this script exits 0.

First run: no baseline exists yet, so this script saves the current results as the
baseline and passes automatically. Every run after that is a real check.

Usage:
    python compare_to_baseline.py --current results/quality.json results/systems.json \
                                   --baseline-dir results/baseline \
                                   --quality-drop-threshold 0.05 \
                                   --latency-regression-threshold 0.20
"""
import argparse
import json
import shutil
import sys
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", nargs="+", required=True, help="Current run's result JSON files")
    ap.add_argument("--baseline-dir", default="results/baseline")
    ap.add_argument("--quality-drop-threshold", type=float, default=0.05,
                     help="Max allowed drop in exact-match accuracy, e.g. 0.05 = 5 points")
    ap.add_argument("--latency-regression-threshold", type=float, default=0.20,
                     help="Max allowed relative increase in p95 latency, e.g. 0.20 = 20% slower")
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    failures = []

    for current_path in args.current:
        current = load(current_path)
        baseline_path = baseline_dir / Path(current_path).name

        if not baseline_path.exists():
            shutil.copy(current_path, baseline_path)
            print(f"No baseline for {current_path} yet -- saved this run as the new baseline. Pass.")
            continue

        baseline = load(str(baseline_path))

        # Quality regression check
        if "accuracy" in current and "accuracy" in baseline:
            drop = baseline["accuracy"] - current["accuracy"]
            if drop > args.quality_drop_threshold:
                failures.append(
                    f"Quality regression in {current_path}: accuracy dropped "
                    f"{drop:.3f} (baseline {baseline['accuracy']:.3f} -> current {current['accuracy']:.3f}), "
                    f"threshold is {args.quality_drop_threshold}"
                )

        # Systems regression check (p95 latency, worst-case concurrency sweep)
        if "sweeps" in current and "sweeps" in baseline:
            for cur_sweep, base_sweep in zip(current["sweeps"], baseline["sweeps"]):
                cur_p95 = cur_sweep["latency_percentiles_s"]["p95"]
                base_p95 = base_sweep["latency_percentiles_s"]["p95"]
                if base_p95 > 0:
                    rel_increase = (cur_p95 - base_p95) / base_p95
                    if rel_increase > args.latency_regression_threshold:
                        failures.append(
                            f"Latency regression in {current_path} at concurrency="
                            f"{cur_sweep['concurrency']}: p95 went from {base_p95:.3f}s to "
                            f"{cur_p95:.3f}s ({rel_increase:.1%} worse), "
                            f"threshold is {args.latency_regression_threshold:.0%}"
                        )

    if failures:
        print("GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("GATE PASSED: no regression past threshold.")
    sys.exit(0)


if __name__ == "__main__":
    main()
EOF_scripts_regression_gate_py

chmod +x scripts/machine_report.sh
python3 -m py_compile evalkit/metrics.py run_quality.py run_ttft_sweep.py scripts/*.py
echo
echo '== tree =='; ls -a; ls evalkit scripts data
echo
echo 'setup done — now:  python3 -c "from evalkit import lm_bpb, eval_generation, bootstrap_ci; print(\"evalkit ok\")"'
