"""Exact compatibility with the original unbatched landmark extraction."""

import hashlib

import numpy as np
import pytest
from scipy import ndimage, signal

import audio_fingerprint
from audio_fingerprint import (
    FFT_SIZE,
    HOP_SIZE,
    SAMPLE_RATE,
    WindowSampler,
    fingerprint,
)


def original_fingerprint(audio: np.ndarray, start_sample: int) -> list[tuple[str, int]]:
    """Retain the version-0.2.0 algorithm as an independent compatibility oracle."""
    _, _, power = signal.spectrogram(
        audio.astype(np.float64) * 32768,
        fs=SAMPLE_RATE,
        window=np.hanning(FFT_SIZE),
        nperseg=FFT_SIZE,
        noverlap=FFT_SIZE - HOP_SIZE,
        detrend=False,
        scaling="density",
        mode="psd",
    )
    db = np.zeros_like(power)
    np.log10(power, out=db, where=power != 0)
    db *= 10
    background = ndimage.binary_erosion(db == 0, structure=np.ones((21, 21)), border_value=1)
    peaks = (ndimage.maximum_filter(db, size=21) == db) != background
    frequencies, frames = np.where(peaks & (db > 10))
    order = np.lexsort((frequencies, frames))
    frequencies, frames = frequencies[order], frames[order]
    result = set()
    for index, (frequency, frame) in enumerate(zip(frequencies, frames, strict=True)):
        for other in range(index + 1, min(index + 5, len(frames))):
            delta = int(frames[other] - frame)
            if 0 <= delta <= 200:
                pair = f"{int(frequency)}|{int(frequencies[other])}|{delta}"
                digest = hashlib.sha1(pair.encode("ascii"), usedforsecurity=False).hexdigest()[:20]
                result.add((digest, int(frame) + start_sample // HOP_SIZE))
    return sorted(result, key=lambda item: (item[1], item[0]))


@pytest.mark.parametrize(
    "count",
    [FFT_SIZE, FFT_SIZE + 63 * HOP_SIZE, FFT_SIZE + 64 * HOP_SIZE + 123, 30 * SAMPLE_RATE],
)
@pytest.mark.parametrize("kind", ["noise", "tone", "gaps", "silence"])
def test_hashes_and_offsets_match_original(count: int, kind: str) -> None:
    """FFT batching must not alter peaks at block boundaries or silence transitions."""
    audio = np.random.default_rng(127).normal(0, 0.1, count).astype(np.float32)
    if kind == "tone":
        audio = (0.5 * np.sin(np.arange(count) * 2 * np.pi * 440 / SAMPLE_RATE)).astype(np.float32)
    elif kind == "gaps":
        audio[count // 3 : 2 * count // 3] = 0
    elif kind == "silence":
        audio[:] = 0
    saved = audio.copy()
    offset = 17 * HOP_SIZE
    assert fingerprint(audio, offset) == original_fingerprint(audio, offset)
    np.testing.assert_array_equal(audio, saved)


def test_fft_intermediates_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """No scipy call may allocate windowed intermediates for the entire 30-second input."""
    original = signal.spectrogram
    sizes = []

    def checked(audio: np.ndarray, **kwargs: object) -> tuple:
        sizes.append(len(audio))
        return original(audio, **kwargs)

    monkeypatch.setattr(signal, "spectrogram", checked)
    fingerprint(np.random.default_rng(9).normal(0, 0.1, 30 * SAMPLE_RATE).astype(np.float32))
    assert len(sizes) > 1
    assert max(sizes) <= FFT_SIZE + 83 * HOP_SIZE


def test_full_track_matches_original(monkeypatch: pytest.MonkeyPatch) -> None:
    """Overlapping windows and the final tail preserve exact original landmarks."""
    audio = np.random.default_rng(91).normal(0, 0.1, 65 * SAMPLE_RATE).astype(np.float32)

    def run() -> list[tuple[str, int]]:
        sampler = WindowSampler(65, SAMPLE_RATE, 1, "pcm_f32le")
        hashes = []
        for part in np.array_split(audio, 97):
            hashes.extend(sampler.feed(part.tobytes()))
        hashes.extend(sampler.feed(b"", last=True))
        assert sampler.position == len(audio)
        assert sampler.completed[-1].start_sample + sampler.completed[-1].sample_count == len(audio)
        return hashes

    actual = run()
    monkeypatch.setattr(audio_fingerprint, "fingerprint", original_fingerprint)
    assert actual == run()


def test_peak_filter_intermediates_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Peak-filter batches include both halos but never the whole spectrogram."""
    original = ndimage.maximum_filter
    widths = []

    def checked(data: np.ndarray, **kwargs: object) -> np.ndarray:
        widths.append(data.shape[1])
        return original(data, **kwargs)

    monkeypatch.setattr(ndimage, "maximum_filter", checked)
    fingerprint(np.random.default_rng(11).normal(0, 0.1, 30 * SAMPLE_RATE).astype(np.float32))
    assert len(widths) > 1
    assert max(widths) <= 64 + 20
