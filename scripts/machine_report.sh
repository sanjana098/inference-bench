#!/usr/bin/env bash
DOCKER_TEST=0
[[ "$1" == "--docker-test" ]] && DOCKER_TEST=1

section() { echo; echo "=== $1 ==="; }
have()    { command -v "$1" >/dev/null 2>&1; }

echo "machine_report generated: $(date)"
echo "hostname: $(hostname)"

section "OS"
[[ -f /etc/os-release ]] && head -4 /etc/os-release || echo "no /etc/os-release"
echo "kernel: $(uname -r)"
have ldd && echo "glibc: $(ldd --version 2>/dev/null | head -1)"

section "GPU"
if have nvidia-smi; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap --format=csv
elif have rocm-smi; then
  rocm-smi --showproductname --showmeminfo vram
else
  echo "no nvidia-smi or rocm-smi found"
fi

section "GPU TOPOLOGY"
if have nvidia-smi; then nvidia-smi topo -m 2>/dev/null || echo "topo not available"; else echo "n/a"; fi

section "CUDA TOOLKIT"
if have nvcc; then nvcc --version | tail -1; else echo "nvcc not installed"; fi

section "CPU"
lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)"

section "RAM"
free -h

section "DISK"
df -h / "$HOME" 2>/dev/null
lsblk -d -o NAME,SIZE,ROTA,TYPE 2>/dev/null

section "DOCKER"
if have docker; then
  docker --version 2>&1
  docker info 2>/dev/null | grep -i -A3 "runtimes" || echo "could not read docker info"
else
  echo "docker not installed"
fi

section "PYTHON"
have python3 && python3 --version || echo "python3 not found"
have pip3 && pip3 --version 2>&1 || echo "pip3 not found"

section "GPU PROCESSES"
if have nvidia-smi; then
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null
else
  echo "n/a"
fi

if [[ $DOCKER_TEST -eq 1 ]]; then
  section "CONTAINERIZED GPU TEST"
  if have docker; then
    docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi 2>&1 \
      || echo "FAILED: containerized GPU access not working"
  else
    echo "docker not installed, skipping"
  fi
fi
