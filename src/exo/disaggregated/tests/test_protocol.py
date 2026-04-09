"""Unit tests for the prefill/decode disaggregation wire protocol.

PR #1776 landed the disaggregated module without any unit tests. These
verify that the binary protocol round-trips correctly for all three
message types (KVChunk, ArraysState, Done) across the supported dtypes
(float16, bfloat16, float32).
"""

from __future__ import annotations

import io

import pytest
import torch

from exo.disaggregated.protocol import (
    ArraysState,
    Done,
    KVChunk,
    read_header,
    read_message,
    write_arrays_state,
    write_done,
    write_header,
    write_kv_chunk,
)


def _roundtrip(
    writer_calls: list[tuple[str, tuple, dict]],
    header: dict[str, object],
) -> list:
    """Run a sequence of writer calls, then read back all messages."""
    buf = io.BytesIO()
    write_header(buf, header)
    writers = {
        "kv": write_kv_chunk,
        "arrays": write_arrays_state,
        "done": write_done,
    }
    for name, args, kwargs in writer_calls:
        writers[name](buf, *args, **kwargs)

    buf.seek(0)
    got_header = read_header(buf)
    assert got_header == header

    results = []
    while True:
        msg = read_message(buf, got_header)
        if msg is None:
            break
        results.append(msg)
    return results


def test_header_roundtrip_simple():
    header = {"dtype": "float16", "model": "test/model", "layers": 32}
    buf = io.BytesIO()
    write_header(buf, header)
    buf.seek(0)
    got = read_header(buf)
    assert got == header


def test_header_roundtrip_unicode():
    header = {"dtype": "bfloat16", "note": "héllo wörld 🚀"}
    buf = io.BytesIO()
    write_header(buf, header)
    buf.seek(0)
    assert read_header(buf) == header


def test_read_header_on_empty_stream_raises():
    buf = io.BytesIO()
    with pytest.raises(ConnectionError):
        read_header(buf)


@pytest.mark.parametrize(
    "dtype_str,torch_dtype",
    [
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("float32", torch.float32),
    ],
)
def test_kv_chunk_roundtrip_3d(dtype_str: str, torch_dtype: torch.dtype):
    """KV chunk with 3D input [num_tokens, n_heads, head_dim]."""
    num_tokens, n_heads, head_dim = 7, 4, 8
    keys = torch.randn(num_tokens, n_heads, head_dim).to(torch_dtype)
    values = torch.randn(num_tokens, n_heads, head_dim).to(torch_dtype)

    messages = _roundtrip(
        [("kv", (3, keys, values), {})],
        header={"dtype": dtype_str},
    )
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, KVChunk)
    assert msg.layer_idx == 3
    assert msg.num_tokens == num_tokens
    assert msg.keys.shape == (num_tokens, n_heads, head_dim)
    assert msg.keys.dtype == torch_dtype
    assert torch.equal(msg.keys, keys)
    assert torch.equal(msg.values, values)


def test_kv_chunk_roundtrip_4d_gets_flattened():
    """4D input [blocks, block_size, n_heads, head_dim] is flattened."""
    blocks, block_size, n_heads, head_dim = 2, 3, 4, 8
    keys = torch.randn(blocks, block_size, n_heads, head_dim).to(torch.float16)
    values = torch.randn(blocks, block_size, n_heads, head_dim).to(torch.float16)

    messages = _roundtrip(
        [("kv", (5, keys, values), {})],
        header={"dtype": "float16"},
    )
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, KVChunk)
    assert msg.num_tokens == blocks * block_size
    assert msg.keys.shape == (blocks * block_size, n_heads, head_dim)
    expected_keys = keys.reshape(-1, n_heads, head_dim)
    assert torch.equal(msg.keys, expected_keys)


def test_done_roundtrip():
    messages = _roundtrip(
        [("done", (4096,), {})],
        header={"dtype": "float16"},
    )
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, Done)
    assert msg.total_tokens == 4096


@pytest.mark.parametrize("dtype_str,torch_dtype", [
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
    ("float32", torch.float32),
])
def test_arrays_state_roundtrip(dtype_str: str, torch_dtype: torch.dtype):
    arr1 = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch_dtype)
    arr2 = torch.ones((4,), dtype=torch_dtype) * 2
    messages = _roundtrip(
        [("arrays", (7, [arr1, arr2]), {})],
        header={"dtype": dtype_str},
    )
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, ArraysState)
    assert msg.layer_idx == 7
    assert len(msg.arrays) == 2
    assert msg.arrays[0].shape == (2, 3)
    assert msg.arrays[1].shape == (4,)
    assert torch.equal(msg.arrays[0], arr1)
    assert torch.equal(msg.arrays[1], arr2)


def test_arrays_state_mixed_dtypes_in_one_message():
    """ArraysState supports per-array dtype annotation."""
    arr_f16 = torch.randn(2, 3).to(torch.float16)
    arr_f32 = torch.randn(2, 3).to(torch.float32)
    messages = _roundtrip(
        [("arrays", (0, [arr_f16, arr_f32]), {})],
        header={"dtype": "float16"},
    )
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, ArraysState)
    assert msg.arrays[0].dtype == torch.float16
    assert msg.arrays[1].dtype == torch.float32
    assert torch.equal(msg.arrays[0], arr_f16)
    assert torch.equal(msg.arrays[1], arr_f32)


def test_multi_message_stream():
    """Realistic sequence: N KV chunks, then Done."""
    keys_per_layer = [torch.randn(3, 4, 8).to(torch.bfloat16) for _ in range(5)]
    values_per_layer = [torch.randn(3, 4, 8).to(torch.bfloat16) for _ in range(5)]

    writer_calls = [
        ("kv", (i, keys_per_layer[i], values_per_layer[i]), {}) for i in range(5)
    ]
    writer_calls.append(("done", (15,), {}))

    messages = _roundtrip(writer_calls, header={"dtype": "bfloat16"})

    assert len(messages) == 6
    for i in range(5):
        msg = messages[i]
        assert isinstance(msg, KVChunk)
        assert msg.layer_idx == i
        assert torch.equal(msg.keys, keys_per_layer[i])
        assert torch.equal(msg.values, values_per_layer[i])
    last = messages[5]
    assert isinstance(last, Done)
    assert last.total_tokens == 15


def test_truncated_stream_raises():
    """Mid-message truncation surfaces ConnectionError."""
    buf = io.BytesIO()
    write_header(buf, {"dtype": "float16"})
    keys = torch.randn(3, 4, 8).to(torch.float16)
    values = torch.randn(3, 4, 8).to(torch.float16)
    write_kv_chunk(buf, 0, keys, values)

    # Truncate the last 100 bytes of the payload
    data = buf.getvalue()
    truncated = io.BytesIO(data[:-100])
    header = read_header(truncated)
    with pytest.raises(ConnectionError):
        read_message(truncated, header)


def test_read_message_returns_none_at_clean_eof():
    """read_message returns None when stream is cleanly exhausted."""
    buf = io.BytesIO()
    write_header(buf, {"dtype": "float16"})
    buf.seek(0)
    header = read_header(buf)
    msg = read_message(buf, header)
    assert msg is None


def test_unknown_message_type_raises():
    """Invalid type byte raises ValueError."""
    buf = io.BytesIO()
    write_header(buf, {"dtype": "float16"})
    buf.write(bytes([0xFF]))  # unknown type byte
    buf.seek(0)
    header = read_header(buf)
    with pytest.raises(ValueError, match="Unknown message type"):
        read_message(buf, header)
