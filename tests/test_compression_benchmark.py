"""Benchmark representations preserve exact hashes and offsets."""

import pytest

from scripts.benchmark_compression import compare, pack, unpack


@pytest.mark.parametrize("delta", [False, True])
def test_binary_round_trip(delta: bool) -> None:
    pairs = [
        ("0" * 20, 0),
        ("f" * 20, 0),
        ("0123456789abcdef0123", 127),
        ("a" * 20, 255),
        ("b" * 20, 38759),
    ]
    assert unpack(pack(pairs, delta), delta) == sorted(pairs)
    assert unpack(pack([], delta), delta) == []


@pytest.mark.parametrize("delta", [False, True])
def test_truncated_record_is_rejected(delta: bool) -> None:
    payload = pack([("a" * 20, 200)], delta)
    with pytest.raises(ValueError, match="Truncated"):
        unpack(payload[:-1], delta)


def test_delta_sorts_unordered_offsets_without_changing_pairs() -> None:
    pairs = [("a" * 20, 2), ("b" * 20, 1)]
    assert unpack(pack(pairs, True), True) == sorted(pairs)


def test_delta_overflow_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        unpack(bytes(10) + b"\xff" * 5 + b"\x01", True)


def test_all_benchmark_variants_round_trip() -> None:
    pairs = [(f"{index:020x}", index // 3) for index in range(100)]
    results = compare(pairs, 1)
    assert len(results) == 12
    assert all(result["bytes"] > 0 for result in results)
