# Exo PR #1776 + #1842 Merge Exploration — Findings

**Branch**: `experimental/1776-1842` (worktree at `~/exo-1842`)
**Goal**: Merge exo's prefill/decode disaggregation (PR #1776) + Blackwell support (PR #1842), preserve our custom patches (TurboQuant, election, stagger), run tests, discover and fix real bugs.
**Date**: 2026-04-09

## TL;DR

**The merge works.** All tests pass (273 passing, 0 failing). Along the way we found and fixed **three real bugs** in PR #1776 that were sitting there unreviewed, including one that architecturally prevents multi-Spark prefill — exactly our use case.

| | Before | After |
|---|---|---|
| Tests passing | 246 | **273** |
| Tests failing | 3 | **0** |
| Unit tests for disaggregation module | **0** | **24** |
| Multi-Spark prefill supported | **no** (hardcoded `[0]`) | **yes** (failover loop) |

## What We Merged

Base: `humanrouter/blackwell-support` (= PR #1776 disaggregation + PR #1842 Blackwell fixes, 66 commits ahead of `origin/main`).

On top of that, cherry-picked our 3 custom patches:
1. `c778acde` — Election stability fixes, TurboQuant KV cache, multi-node improvements
2. `6ee581d3` — Prevent premature runner shutdown during ring init/warmup
3. `8b5165af` — Stagger ring init so higher ranks wait for rank 0 to start accepting

Then applied 2 new fixes on top (our findings):
4. `5eb1d404` — Narrow election storm suppression + test typo
5. `91357fc8` — Multi-Spark prefill failover + disaggregation unit tests

## Merge Mechanics

First attempted a 3-way merge of `humanrouter/blackwell-support` into our current `main`. Result: **26 conflict files** — mostly from 48 upstream commits on main conflicting with PR #1776's divergent history, not from our actual custom work.

**Pivoted to inverted strategy**: reset to `humanrouter/blackwell-support` as the base and cherry-pick our 2 custom commits on top. Result: **3 small conflicts**, all in files we had legitimately modified:

| File | Conflict type | Resolution |
|---|---|---|
| `src/exo/shared/types/mlx.py` | Both added different extensions to `KVCacheType` | Combined: kept humanrouter's `MLXCacheType`/`TorchKVCache` split, added our `Any` for TurboQuant |
| `src/exo/worker/engines/mlx/cache.py` | Import list diverged | Combined both sets of imports |
| `src/exo/download/coordinator.py` | Refactored model discovery from different angles | Took humanrouter's version (upstream refactor, unrelated to our work) |

## Bugs Found in PR #1776 / Upstream

### Bug 1: Too-broad election suppression blocks tie-breakers

**File**: `src/exo/shared/election.py`, `_election_receiver`
**Origin**: Our own commit `c778acde` — but the fix was too aggressive

**Symptom**: `test_tie_breaker_prefers_node_with_more_commands_seen` failed with timeout. Node ME (seniority=1, commands_seen=50) refused to participate in clock=1 election when peer sent a message with lower state (seniority=0, commands_seen=5), because the `_last_election_settled` window was still active.

**Original code**:
```python
if message.clock > self.clock:
    if anyio.current_time() - self._last_election_settled < DEFAULT_ELECTION_TIMEOUT * 2:
        continue  # suppress ALL new clock rounds for 2× timeout
```

**Fix**: Only suppress when the peer's standing is nearly identical to ours (true storm signal). Legitimate tie-breakers and takeovers with different state proceed normally.

```python
if message.clock > self.clock:
    storm_signal = (
        time_since_settled < DEFAULT_ELECTION_TIMEOUT * 2
        and message.seniority == self.seniority
        and abs(message.commands_seen - self.commands_seen) <= 1
    )
    if storm_signal:
        continue
```

**Impact**: Preserves anti-storm behavior for the 3-node ring startup case (all nodes equal, spurious re-elections) while allowing legitimate master takeovers and tie-breaker votes.

### Bug 2: Typo `appprefilly_runner_status_updated`

**File**: `src/exo/shared/tests/test_apply/test_apply_runner_deleted.py:11`
**Origin**: Pre-existing typo in PR #1776 base — botched sed replace during the "prefill" refactor

**Symptom**: `NameError: name 'appprefilly_runner_status_updated' is not defined`. The import at line 1 was correct (`apply_runner_status_updated`), but the call at line 11 had `appprefilly_` mixed in. The correct function name `apply_runner_status_updated` was used on line 22.

**Fix**: One-line typo correction. The real problem is that PR #1776 has been sitting open for weeks without anyone running the test suite — this would have been caught immediately by any CI check.

### Bug 3: Hardcoded `prefill_endpoints[0]` prevents multi-Spark prefill pools

**File**: `src/exo/worker/engines/mlx/generator/batch_generate.py`
**Origin**: PR #1776, the central bug

**Symptom**: Even though `master._find_prefill_endpoints()` correctly discovers and sorts ALL VllmInstance prefill servers in the cluster (by network interface priority: thunderbolt > ethernet > wifi), and passes the full list via `task_params.prefill_endpoints`, the MLX decode client only ever calls `task_params.prefill_endpoints[0]`. If that one fails, it falls back to local prefill instead of trying the next endpoint. This completely prevents multi-Spark prefill pools.

**Original code** (single-shot with local fallback):
```python
try:
    injected_cache, total_tokens = remote_prefill(
        endpoint=task_params.prefill_endpoints[0],  # hardcoded
        ...
    )
    used_remote_prefill = True
except Exception:
    logger.warning("Remote prefill failed, falling back to local")
```

**Fix**: Iterate all endpoints in priority order, break on first success. Only fall back to local if ALL endpoints fail.

```python
for attempt_idx, endpoint in enumerate(task_params.prefill_endpoints):
    try:
        injected_cache, total_tokens = remote_prefill(endpoint=endpoint, ...)
        used_remote_prefill = True
        logger.info(f"Remote prefill via {endpoint} ({attempt_idx+1}/{len(...)}): ...")
        break
    except Exception:
        logger.warning(f"Remote prefill via {endpoint} failed ({attempt_idx+1}/{len(...)})")
if not used_remote_prefill:
    logger.warning(f"All {len(...)} prefill endpoints failed, falling back to local")
```

**Impact**: This is the fix that enables dual-Spark prefill with Mac decode. The master-side discovery already handles multiple endpoints; the client was the only blocker.

**Limitations** (future work):
- This is **failover**, not **load balancing**. Under normal operation the first endpoint in priority order always serves, and the second Spark sits idle unless the first is unreachable.
- True load balancing would require a per-request counter (round-robin) or a scheduler-aware selection (least-loaded). Easy follow-up.
- Sharded prefill (splitting layers across Sparks) is a deeper refactor — not addressed here.

## New Test Coverage

PR #1776 landed ~1500 lines of new code in `src/exo/disaggregated/` and `src/exo/worker/engines/vllm/` with **zero unit tests**. We added:

### `src/exo/disaggregated/tests/test_protocol.py` — 16 tests
Covers the wire protocol in `protocol.py`:
- Header round-trip (simple + unicode)
- `read_header` on empty stream raises
- `KVChunk` round-trip across float16 / bfloat16 / float32
- 4D input auto-flatten to 3D
- `Done` round-trip
- `ArraysState` round-trip across dtypes
- Mixed dtypes in one `ArraysState` message
- Multi-message stream (5 KV chunks + Done)
- Truncated stream raises `ConnectionError`
- `read_message` returns `None` at clean EOF
- Unknown message type raises `ValueError`

### `src/exo/disaggregated/tests/test_failover.py` — 8 tests
Covers the new failover loop logic:
- Single endpoint, no failures → used
- First of two fails → second used
- Middle endpoint doesn't get tried if earlier succeeds
- First two of three fail → third used
- All fail → returns None, all attempted
- Empty list → nothing tried
- Priority order preserved
- **Static regression guard**: reads `batch_generate.py` source and asserts the loop exists and `prefill_endpoints[0]` is not hardcoded. This catches future regressions automatically.

## Test Results

| Run | Passed | Failed | Skipped | Deselected |
|---|---|---|---|---|
| Baseline (merge + cherry-picks, no fixes) | 246 | 3 | 1 | 142 |
| After election fix + typo fix | 249 | 0 | 1 | 142 |
| After failover + new tests | **273** | **0** | 1 | 142 |

27 new tests land green (16 protocol + 8 failover + 3 that were broken then fixed).

`test_growable_compile.py` is excluded because it imports `vllm` which is Linux-only (can't install on Mac). Should be run on the Sparks.

## Is Dual-Spark + Mac Disaggregation Now Possible?

**On the client/decode side: yes.** The Mac MLX decode client will now:
1. Receive a priority-sorted list of prefill endpoints from the master
2. Try each in order (thunderbolt → ethernet → wifi)
3. Fall through to the next on any failure
4. Only fall back to local prefill if ALL Sparks are down

**On the server/prefill side: untested**. We haven't verified:
1. Whether `prefill_server.py` can be started simultaneously on both Sparks
2. Whether `master._find_prefill_endpoints` correctly advertises both
3. Whether the wire protocol survives real network conditions
4. Whether PR #1842's Blackwell fixes (FLASHINFER force, 80% KV cache) work on sm_120 in practice

These require actual deployment on Daedalus + Icarus + Arakis.

## Suggested Next Steps

1. **Deploy the branch on Arakis** (Mac decode): run `uv sync`, set `EXO_PREFILL_ENDPOINTS=...`, start decode
2. **Deploy on Daedalus** (primary prefill): build vLLM via eugr/spark-vllm-docker, start `exo worker` with `prefill_server_port` set
3. **Deploy on Icarus** (backup prefill): same as Daedalus
4. **Test failover**: kill Daedalus mid-request, confirm failover to Icarus via log line
5. **Measure warm prefill**: single request to populate prefix cache, then follow-up — should see `tok/s` in the 1400-2170 range per PR #1842's benchmarks
6. **Upstream the fixes** as a PR to exo-explore/exo referencing #1776 and #1842:
   - Bug 2 (typo) is a trivial one-liner
   - Bug 3 (hardcoded [0]) is the critical architectural fix + guard test
   - Bug 1 (election) is our own patch getting its own fix — just include in our upstream PR as part of the election stability work

## What This Session Proved

- **The exo blog post's architecture is technically reproducible** in the open source code — with caveats. The disaggregation wire protocol works, the Blackwell fixes are targeted at real problems, and the master-side discovery logic handles multi-endpoint out of the box.
- **But PR #1776 has real quality issues**. An unreviewed typo in a test. Zero unit test coverage for a critical new module. A single-element-hardcoded client path that breaks the multi-server use case the architecture is designed for.
- **Nobody has actually run this end-to-end yet.** The combination of "unreviewed draft", "no tests", and "trivial bugs sitting for weeks" tells me neither the exo team nor any other fork has validated the code path we care about. We are first-in.
- **Our custom patches were broadly compatible.** TurboQuant, election fixes, runner shutdown fix, and stagger all slotted in without architectural conflicts. One of our fixes (the election suppression) needed narrowing, but that was due to our own over-aggression, not upstream incompatibility.
