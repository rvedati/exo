# Deployment Runbook — Experimental Branch `1776-1842`

Deployment guide for exo with PR #1776 (prefill/decode disaggregation) + PR #1842 (Blackwell support) across 2x DGX Spark + 1x Mac Studio.

## Current Status (2026-04-09)

**Code state**: Branch is merged, 273 unit tests pass on macOS, 40 key tests pass on Linux CUDA, pushed to `github.com/rvedati/exo:experimental/1776-1842`.

**Deployment state**: NOT yet end-to-end tested. The Arakis decode side is verified by unit tests. The Spark prefill side requires a vLLM source build that was blocked this session by a `uv sync` misconfiguration (documented below).

**Running baseline**: dev-branch ring exo is still running on Arakis (EXO.app PID 38534) + Daedalus (PID 95509 from `~/exo-app`). **Do not kill** without planning — this is the working 2-node ring from `2026-04-09 02:10`.

## Critical Gotchas

### 1. `uv sync` on Linux Sparks will UNINSTALL your CUDA stack

The CUDA dependencies are in `[project.optional-dependencies.cuda]`:
```toml
cuda = [
  "torch>=2.10.0; sys_platform == 'linux'",
  "vllm>=0.13.0; sys_platform == 'linux'",
  "mlx-cuda-13==0.30.6; sys_platform == 'linux'",
  "fastsafetensors>=0.1.10; sys_platform == 'linux'",
]
```

Plain `uv sync` does not install optional extras and will **actively uninstall** them if they're present from manual `uv pip install`. Always use:

```bash
uv sync --extra cuda
```

This was discovered the hard way after a plain `uv sync` on both Sparks removed torch, triton, mlx-cuda-13, and ~20 CUDA libraries in ~1 second, then exited without building vLLM.

### 2. vLLM builds from source (not a wheel)

`pyproject.toml` pins a custom fork + no-binary + no-build-isolation:

```toml
[tool.uv.sources]
vllm = { git = "https://github.com/hmellor/vllm.git", branch = "transformers-v5" }

[tool.uv]
no-binary-package = ["vllm"]
no-build-isolation-package = ["vllm"]
```

First sync on a fresh Spark will compile vLLM from source. Per `feedback_spark_builds` memory:
- Set `MAX_JOBS=12` (not 16 — saturates the 20-core Spark so bad SSH becomes unresponsive)
- Set `NVCC_THREADS=2`
- Drop caches before start: `sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`
- Verify swap is off: `free -h` should show `Swap: 0B`
- Expected build time: 30-60 minutes on Spark
- Watch for OOM — earlyoom should protect sshd

### 3. Dashboard assets stub

Import-time check in `src/exo/shared/constants.py` calls `find_dashboard()` which looks for `dashboard/build/index.html`. For test/runtime on hosts where you haven't run `npm install && npm run build`, create a stub:

```bash
mkdir -p dashboard/build
echo '<html><body>stub</body></html>' > dashboard/build/index.html
```

### 4. MTU 9000 + NIC IRQ pinning required on Sparks

See `~/.claude/projects/-Users-rikv-ClaudeHomeLabOrchestrator/memory/project_jumbo_frames_10gbe.md`. Already persistent via:
- `/etc/netplan/50-lan-mtu.yaml` (MTU 9000)
- `/etc/systemd/system/nic-irq-bigcores.service` (pin IRQs to Cortex-X925 big cores 5-9,15-19)

Without big-core pinning, Daedalus hits 7 Gbit/s instead of line-rate 9.9 Gbit/s because kernel drops NIC IRQ on a Cortex-A725 little core.

## Pre-Flight Checklist

```bash
# On each Spark (Daedalus, Icarus):
ssh rikv@192.168.50.4
uv --version                                 # should be present
cat /sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_max_freq  # 3900000 = big core
ip link show enP7s7 | grep mtu               # should show mtu 9000
sudo systemctl status nic-irq-bigcores       # should be active
free -h                                       # swap should be 0B
nvidia-smi --query-gpu=name,driver_version --format=csv  # GB10, 590.48.01
systemctl is-active earlyoom || echo "WARN: earlyoom not running"

# On Arakis:
ifconfig en0 | grep mtu                       # should show mtu 9000
launchctl print system/com.homelab.mtu | grep "last exit code"  # should be 0
```

## Deployment Sequence

### Phase 1: Build vLLM on both Sparks (PARALLEL)

```bash
# Daedalus
ssh rikv@192.168.50.4
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
cd ~/exo-1842   # or clone from github.com/rvedati/exo -b experimental/1776-1842
export PATH=$HOME/.local/bin:$PATH
export MAX_JOBS=12 NVCC_THREADS=2 CMAKE_BUILD_PARALLEL_LEVEL=12 CUDA_ARCH_LIST=12.1
nohup uv sync --extra cuda > /tmp/vllm-build.log 2>&1 &
echo $! > /tmp/vllm-build.pid

# Same on Icarus (192.168.50.5)
```

Monitor with `tail -f /tmp/vllm-build.log`. Expect 30-60 min. If it fails with ABI mismatch, check that `torch` version matches the PyTorch cu130 index.

### Phase 2: Sanity check the venvs

```bash
# Both Sparks:
uv run python -c "
import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
import vllm; print('vllm', vllm.__version__)
import mlx.core as mx; print('mlx device', mx.default_device())
import exo.disaggregated.protocol
import exo.worker.engines.vllm.vllm_generator
print('exo disaggregation imports OK')
"
```

Expected output:
```
torch 2.11.0+cu130 cuda True
vllm 0.14.x (or whatever hmellor's branch resolves to)
mlx device Device(gpu, 0)
exo disaggregation imports OK
```

### Phase 3: Download Qwen3.5-27B

Per BLACKWELL.md, the documented test model:

```bash
# On each Spark (needs ~54GB for bf16):
ssh rikv@192.168.50.4
uv run huggingface-cli download Qwen/Qwen3.5-27B

# On Arakis (needs mxfp8 variant, ~13.5GB):
huggingface-cli download mlx-community/Qwen3.5-27B-mxfp8
```

Alternatively, start with already-downloaded Nemotron-Nano-4B on Daedalus for a smaller smoke test (but it's missing the Qwen3.5-27B model card that PR #1842 added).

### Phase 4: Stop existing ring exo

**DESTRUCTIVE — coordinate with user first.**

```bash
# Arakis:
osascript -e 'tell application "EXO" to quit'

# Daedalus:
ssh rikv@192.168.50.4 "pkill -f 'uv run exo' && pkill -f '.venv/bin/exo'"

# Verify ports free:
ssh rikv@192.168.50.4 "ss -tlnp | grep -E ':52334|:52415|:8900' || echo clear"
```

### Phase 5: Start disaggregation

Per BLACKWELL.md, just run `uv run exo` on each machine and select the right model in each dashboard:

```bash
# Daedalus (Spark 1 — prefill):
ssh rikv@192.168.50.4
cd ~/exo-1842
CUDA_FORCE_PTX_JIT=1 nohup uv run exo > ~/exo.log 2>&1 &
# In dashboard at http://192.168.50.4:52415, select Qwen/Qwen3.5-27B → Vllm / Pipeline

# Icarus (Spark 2 — prefill):
ssh rikv@192.168.50.5
cd ~/exo-1842
CUDA_FORCE_PTX_JIT=1 nohup uv run exo > ~/exo.log 2>&1 &
# In dashboard, select Qwen/Qwen3.5-27B → Vllm / Pipeline

# Arakis (Mac — decode):
cd ~/exo-1842
uv run exo
# In dashboard, select mlx-community/Qwen3.5-27B-mxfp8 → MlxRing / Pipeline / 1 node
```

The master on Arakis will discover both Spark prefill endpoints via `_find_prefill_endpoints()` and pass them through `task_params.prefill_endpoints` to the MLX decode client.

### Phase 6: First inference request

Submit a chat completion with **>2000 uncached tokens** to trigger remote prefill:

```bash
curl -X POST http://192.168.50.2:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-27B-mxfp8",
    "messages": [
      {"role": "system", "content": "<long system prompt, 2000+ tokens>"},
      {"role": "user", "content": "what is 2+2"}
    ],
    "max_tokens": 50
  }'
```

Expected log lines on Arakis:
```
Remote prefill via 192.168.50.4:8900 (1/2): 2467 tokens at 354 tok/s
```

First call will be cold (~8s TTFT per PR #1842 benchmarks). Follow-up calls in the same conversation benefit from prefix caching.

### Phase 7: Failover test

Kill Daedalus's exo mid-request, verify decode client fails over to Icarus:

```bash
ssh rikv@192.168.50.4 "pkill -f 'uv run exo'"
# Immediately submit another prompt
# Expected log: "Remote prefill via 192.168.50.4:8900 failed (1/2)"
# Then:        "Remote prefill via 192.168.50.5:8900 (2/2): ... tokens at ... tok/s"
```

This validates our `feat: multi-Spark prefill failover` commit end-to-end.

## Performance Targets (from PR #1842)

| Metric | Target | Notes |
|---|---|---|
| Cold prefill (1st prompt, 2467 tok) | 354 tok/s | 1st call always slower |
| Warm prefill (follow-up, 2700 new) | 1,400–2,170 tok/s | Prefix cache accelerates |
| Network transfer (~2700 tok KV) | ~2.9s | Over 10 GbE MTU 9000 |
| MLX decode (Mac) | 18.7 tok/s | mxfp8 |
| Follow-up TTFT | 2.3–3.7s | New tokens dependent |

If you're significantly below these, check:
1. MTU is actually 9000 end-to-end (`ping -c 3 -D -s 8972 192.168.50.4`)
2. NIC IRQs are pinned to big cores (`systemctl status nic-irq-bigcores`)
3. vLLM is using FLASHINFER (`grep FLASHINFER ~/exo.log`)
4. GPU memory utilization is ~80% (`nvidia-smi`)

## What This Branch Fixes Over Upstream PR #1776/#1842

1. **Election storm suppression** narrowed to only fire when state is nearly identical (restored tie-breaker semantics)
2. **`appprefilly_runner_status_updated` typo** in `test_apply_runner_deleted.py:11` — PR #1776 had this unreviewed bug
3. **Multi-Spark prefill failover** — PR #1776 hardcoded `prefill_endpoints[0]`. Our fix iterates all endpoints, enabling the dual-Spark pool the architecture is designed for but never shipped.
4. **24 new unit tests** for the disaggregated module (16 protocol, 8 failover). PR #1776 landed ~1500 lines of new code with zero tests.

All changes are minimal and targeted for upstream PR submission to `exo-explore/exo`.

## Deferred Work (Next Session)

- Phase 1-7 above (actual end-to-end deployment with vLLM build)
- Consider cherry-picking `humanrouter/alexcheema/mlx-distributed-transfer` if dual-Spark TP hits memory issues
- Consider Skulk's OptiQ KV backend if TurboQuant proves insufficient
- Upstream PR with the 3 bug fixes
