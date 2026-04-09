"""Unit test for the prefill failover loop in MLX batch_generate.

This doesn't exercise the full batch_generate pipeline (which requires
a loaded model, MLX runtime, and tokenizer). Instead it tests the
isolated failover logic: given a list of endpoints, the client should
try each in order and stop on the first success.

Before this test existed the client hardcoded `prefill_endpoints[0]`,
preventing multi-Spark prefill pools entirely.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _run_failover(endpoints: list[str], fail_on: set[str]) -> tuple[str | None, list[str]]:
    """Reimplementation of the failover loop for isolated testing.

    Returns (successful_endpoint_or_none, list_of_endpoints_tried_in_order).
    Mirrors the logic in src/exo/worker/engines/mlx/generator/batch_generate.py.
    """
    used_remote_prefill = False
    successful_endpoint: str | None = None
    attempted: list[str] = []
    for endpoint in endpoints:
        attempted.append(endpoint)
        try:
            if endpoint in fail_on:
                raise ConnectionRefusedError(f"mock failure on {endpoint}")
            used_remote_prefill = True
            successful_endpoint = endpoint
            break
        except Exception:
            continue
    return successful_endpoint, attempted


def test_failover_single_endpoint_success():
    """One endpoint, no failures → used."""
    got, attempted = _run_failover(["spark1:8900"], set())
    assert got == "spark1:8900"
    assert attempted == ["spark1:8900"]


def test_failover_first_endpoint_fails_second_succeeds():
    """Two endpoints, first fails → second used."""
    got, attempted = _run_failover(
        ["spark1:8900", "spark2:8900"],
        fail_on={"spark1:8900"},
    )
    assert got == "spark2:8900"
    assert attempted == ["spark1:8900", "spark2:8900"]


def test_failover_middle_endpoint_fails():
    """Three endpoints, middle fails → first succeeds, second never tried."""
    got, attempted = _run_failover(
        ["spark1:8900", "spark2:8900", "spark3:8900"],
        fail_on={"spark2:8900"},
    )
    # First succeeds, loop breaks before trying middle
    assert got == "spark1:8900"
    assert attempted == ["spark1:8900"]


def test_failover_first_two_fail_third_succeeds():
    """Three endpoints, first two fail → third used."""
    got, attempted = _run_failover(
        ["spark1:8900", "spark2:8900", "spark3:8900"],
        fail_on={"spark1:8900", "spark2:8900"},
    )
    assert got == "spark3:8900"
    assert attempted == ["spark1:8900", "spark2:8900", "spark3:8900"]


def test_failover_all_endpoints_fail():
    """All fail → None returned, all attempted."""
    endpoints = ["spark1:8900", "spark2:8900", "spark3:8900"]
    got, attempted = _run_failover(endpoints, fail_on=set(endpoints))
    assert got is None
    assert attempted == endpoints


def test_failover_empty_list():
    """Empty list → nothing tried."""
    got, attempted = _run_failover([], set())
    assert got is None
    assert attempted == []


def test_failover_priority_order_preserved():
    """The order of the input list is honored (matches master priority sort)."""
    endpoints = ["tb:8900", "eth:8900", "wifi:8900"]
    got, attempted = _run_failover(endpoints, fail_on={"tb:8900", "eth:8900"})
    assert got == "wifi:8900"
    # All three attempted in priority order
    assert attempted == endpoints


def test_batch_generate_has_failover_loop():
    """Static check: confirms the batch_generate.py file contains the
    failover loop structure and does not hardcode prefill_endpoints[0].

    This guards against regressions back to the original bug.
    """
    from pathlib import Path

    # From src/exo/disaggregated/tests/test_failover.py we need
    # src/exo/worker/engines/mlx/generator/batch_generate.py
    # parents[2] = src/exo
    batch_generate_path = (
        Path(__file__).resolve().parents[2]
        / "worker"
        / "engines"
        / "mlx"
        / "generator"
        / "batch_generate.py"
    )
    source = batch_generate_path.read_text()

    # The loop must exist
    assert "for attempt_idx, endpoint in enumerate(task_params.prefill_endpoints)" in source, (
        "Failover loop missing from batch_generate.py — regressed to single-endpoint "
        "behavior. Multi-Spark prefill requires iterating all endpoints."
    )

    # The old hardcoded indexing must be gone
    assert "task_params.prefill_endpoints[0]" not in source, (
        "batch_generate.py still hardcodes prefill_endpoints[0]. This prevents "
        "failover to subsequent endpoints and breaks multi-Spark prefill pools."
    )
