"""Tests of window coverage, PCM decoding, and offset-stable landmark fingerprints."""

from collections import Counter

import numpy as np
import pytest

from music_assistant_fingerprint import (
    FFT_SIZE,
    HOP_SIZE,
    MAX_DURATION,
    SAMPLE_RATE,
    WINDOW_SECONDS,
    WINDOW_STRIDE,
    WindowSampler,
    decode_pcm,
    fingerprint,
    plan_windows,
)


def test_windows_cover_the_whole_track_with_overlap() -> None:
    windows = plan_windows(180)
    assert len(windows) > 3
    assert windows[0].start_sample == 0
    assert windows[-1].start_sample + windows[-1].sample_count == 180 * SAMPLE_RATE
    assert all(window.start_sample % HOP_SIZE == 0 for window in windows)
    assert all(
        left.start_sample < right.start_sample < left.start_sample + left.sample_count
        for left, right in zip(windows, windows[1:], strict=False)
    )


@pytest.mark.parametrize("duration", [FFT_SIZE / SAMPLE_RATE, 10, 30, 45, 60, 90, MAX_DURATION])
def test_short_and_maximum_track_windows(duration: float) -> None:
    windows = plan_windows(duration)
    assert windows[-1].start_sample + windows[-1].sample_count == round(duration * SAMPLE_RATE)
    assert all(
        FFT_SIZE <= window.sample_count <= WINDOW_SECONDS * SAMPLE_RATE for window in windows
    )
    assert all(
        left.start_sample + left.sample_count > right.start_sample
        for left, right in zip(windows, windows[1:], strict=False)
    )


@pytest.mark.parametrize("duration", [0, -1, 0.000001, 0.01, float("nan"), float("inf"), 1801])
def test_invalid_duration(duration: float) -> None:
    with pytest.raises(ValueError):
        plan_windows(duration)


def _audio(seconds: int = 5) -> np.ndarray:
    rng = np.random.default_rng(42)
    audio = rng.normal(0, 0.1, SAMPLE_RATE * seconds).astype(np.float32)
    return np.clip(audio, -1, 1)


def test_hashes_keep_absolute_offsets() -> None:
    audio = _audio()
    first = fingerprint(audio)
    shifted = fingerprint(audio, 100 * HOP_SIZE)
    assert len(first) > 50
    assert shifted == [(digest, offset + 100) for digest, offset in first]
    assert len(set(first)) == len(first)


def test_chunking_does_not_change_fingerprints() -> None:
    audio = _audio()
    sampler = WindowSampler(5, SAMPLE_RATE, 1, "pcm_f32le")
    actual = []
    for chunk in np.array_split(audio, 17):
        actual.extend(sampler.feed(chunk.tobytes()))
    actual.extend(sampler.feed(b"", last=True))
    assert actual == fingerprint(audio)
    assert sampler.completed == list(plan_windows(5))


def test_cancelled_prefix_does_not_complete_a_window() -> None:
    sampler = WindowSampler(180, SAMPLE_RATE, 1, "pcm_f32le")
    assert sampler.feed(_audio().tobytes()[: SAMPLE_RATE * 4]) == []
    with pytest.raises(ValueError, match="track end"):
        sampler.feed(b"", last=True)
    assert sampler.completed == []


@pytest.mark.parametrize(
    "encoding,raw",
    [
        ("pcm_s16le", np.array([16384, -16384], dtype="<i2").tobytes()),
        ("pcm_s32le", np.array([1073741824, -1073741824], dtype="<i4").tobytes()),
        ("pcm_s24le", bytes([0, 0, 64, 0, 0, 192])),
        ("pcm_f32le", np.array([0.5, -0.5], dtype="<f4").tobytes()),
        ("pcm_f64le", np.array([0.5, -0.5], dtype="<f8").tobytes()),
    ],
)
def test_pcm_encodings(encoding: str, raw: bytes) -> None:
    np.testing.assert_array_equal(decode_pcm(raw, encoding, 1), [0.5, -0.5])
    np.testing.assert_array_equal(decode_pcm(raw, encoding, 2), [0])


def test_pcm_rejects_partial_frames() -> None:
    with pytest.raises(ValueError, match="complete frames"):
        decode_pcm(b"\0", "pcm_s16le", 2)


def test_silence_has_no_fingerprints() -> None:
    assert fingerprint(np.zeros(SAMPLE_RATE, dtype=np.float32)) == []


def test_noisy_excerpt_matches_reference_offset() -> None:
    audio = _audio(12)
    reference = fingerprint(audio)
    start = 50 * HOP_SIZE
    excerpt = audio[start : start + SAMPLE_RATE * 5].copy()
    excerpt += np.random.default_rng(7).normal(0, 0.002, len(excerpt)).astype(np.float32)
    query = fingerprint(excerpt)
    indexed: dict[str, list[int]] = {}
    for digest, offset in reference:
        indexed.setdefault(digest, []).append(offset)
    votes = Counter(
        offset - query_offset
        for digest, query_offset in query
        for offset in indexed.get(digest, [])
    )
    assert votes.most_common(1)[0][0] == 50
    assert votes[50] >= 50


def test_resampling_is_independent_of_chunk_boundaries() -> None:
    source_rate = 48000
    audio = np.random.default_rng(3).normal(0, 0.1, source_rate * 5).astype(np.float32)
    results = []
    for chunk_count in (5, 23):
        sampler = WindowSampler(5, source_rate, 1, "pcm_f32le")
        hashes = []
        for part in np.array_split(audio, chunk_count):
            hashes.extend(sampler.feed(part.tobytes()))
        hashes.extend(sampler.feed(b"", last=True))
        assert len(sampler.completed) == 1
        assert sampler.position == SAMPLE_RATE * 5
        results.append(hashes)
    assert results[0] == results[1]


def test_full_track_chunking_and_bounded_buffer() -> None:
    """Every passage is processed with bounded PCM and chunk-independent landmarks."""
    audio = _audio(65)
    results = []
    for chunk_count in (65, 211):
        sampler = WindowSampler(65, SAMPLE_RATE, 1, "pcm_f32le")
        actual = []
        for part in np.array_split(audio, chunk_count):
            actual.extend(sampler.feed(part.tobytes()))
            assert sampler._count <= (WINDOW_SECONDS + 2) * SAMPLE_RATE
        actual.extend(sampler.feed(b"", last=True))
        assert sampler.completed == list(plan_windows(65))
        assert len(set(actual)) == len(actual)
        assert max(offset for _, offset in actual) > 60 * SAMPLE_RATE // HOP_SIZE
        assert sampler._parts == []
        results.append(actual)
    assert results[0] == results[1]


@pytest.mark.parametrize("difference", [-0.3, 0.3])
def test_actual_pcm_tail_is_not_discarded_or_padded(difference: float) -> None:
    sampler = WindowSampler(5 + difference, SAMPLE_RATE, 1, "pcm_f32le")
    hashes = []
    for part in np.array_split(_audio(), 5):
        hashes.extend(sampler.feed(part.tobytes()))
    hashes.extend(sampler.feed(b"", last=True))
    assert hashes == fingerprint(_audio())
    assert sampler.completed[-1].sample_count == 5 * SAMPLE_RATE


def test_exact_maximum_duration_and_tiny_planned_tail() -> None:
    for samples in (MAX_DURATION * SAMPLE_RATE, WINDOW_STRIDE + 10, 30 * SAMPLE_RATE + 1):
        windows = plan_windows(samples / SAMPLE_RATE)
        assert windows[-1].start_sample + windows[-1].sample_count == samples
        assert windows[-1].sample_count >= FFT_SIZE
