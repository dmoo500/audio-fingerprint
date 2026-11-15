"""Bounded landmark fingerprint extraction without database or audio-file dependencies."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import soxr
from numpy.typing import NDArray
from scipy import ndimage, signal

LEGACY_FORMAT_ID = "ma-dejavu-landmarks-v1"
FORMAT_ID = "ma-dejavu-landmarks-v2"
SAMPLE_RATE = 44100
FFT_SIZE = 4096
HOP_SIZE = 2048
WINDOW_SECONDS = 30
WINDOW_STRIDE = (20 * SAMPLE_RATE // HOP_SIZE) * HOP_SIZE
MAX_DURATION = 1800
WINDOW_COUNT = math.ceil(MAX_DURATION * SAMPLE_RATE / WINDOW_STRIDE)
MAX_HASHES = 100_000
MAX_TRACK_HASHES = 1_000_000
MAX_DOCUMENT_BYTES = 36_000_000

Fingerprint = tuple[str, int]


@dataclass(frozen=True)
class Window:
    """A reference window at an absolute sample position in the normalized audio."""

    start_sample: int
    sample_count: int


def plan_windows(duration: float) -> tuple[Window, ...]:
    """
    Cover a finite track with bounded overlapping, hop-aligned windows.

    :param duration: Track duration in seconds, at most MAX_DURATION.
    """
    if not math.isfinite(duration) or not 0 < duration <= MAX_DURATION:
        raise ValueError(f"Duration must be positive and no more than {MAX_DURATION} seconds")
    total = round(duration * SAMPLE_RATE)
    if total < FFT_SIZE:
        raise ValueError("Duration must cover at least one FFT frame")
    size = min(total, (WINDOW_SECONDS * SAMPLE_RATE // HOP_SIZE) * HOP_SIZE)
    windows = []
    start = 0
    while True:
        window = Window(start, min(size, total - start))
        windows.append(window)
        if start + window.sample_count == total:
            return tuple(windows)
        start += WINDOW_STRIDE


def fingerprint(samples: NDArray[np.float32], start_sample: int = 0) -> list[Fingerprint]:
    """
    Extract Dejavu-style frequency-pair hashes from normalized mono audio.

    :param samples: Finite mono samples in [-1, 1] at SAMPLE_RATE, at most 30 seconds.
    :param start_sample: Absolute sample offset, aligned to HOP_SIZE.
    """
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1 or not FFT_SIZE <= audio.size <= WINDOW_SECONDS * SAMPLE_RATE:
        raise ValueError("Expected between one FFT frame and 30 seconds of mono audio")
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1.0:
        raise ValueError("Audio samples must be finite and normalized to [-1, 1]")
    if start_sample < 0 or start_sample % HOP_SIZE:
        raise ValueError("Start offset must be nonnegative and aligned to the FFT hop")
    # Dejavu's amplitude threshold is defined for signed 16-bit PCM, not unit floats.
    frame_count = (audio.size - FFT_SIZE) // HOP_SIZE + 1
    hann = np.hanning(FFT_SIZE)
    positions = []
    # The halo preserves the full-window 21x21 neighborhood while bounding every
    # spectral intermediate. Eroded zero-background pixels cannot pass >10 dB.
    for first in range(0, frame_count, 64):
        last = min(first + 64, frame_count)
        left, right = max(0, first - 10), min(frame_count, last + 10)
        count = right - left
        start = left * HOP_SIZE
        scaled = audio[start : start + FFT_SIZE + (count - 1) * HOP_SIZE].astype(np.float64)
        scaled *= 32768
        _, _, power = signal.spectrogram(
            scaled,
            fs=SAMPLE_RATE,
            window=hann,
            nperseg=FFT_SIZE,
            noverlap=FFT_SIZE - HOP_SIZE,
            detrend=False,
            scaling="density",
            mode="psd",
        )
        del scaled
        np.log10(power, out=power, where=power != 0)
        power *= 10
        maximum = ndimage.maximum_filter(power, size=21)
        center = power[:, first - left : last - left]
        peaks = (maximum[:, first - left : last - left] == center) & (center > 10)
        frequencies, frames = np.where(peaks)
        positions.append((frequencies, frames + first))
        del maximum, center, peaks, power
    frequencies = np.concatenate([frequency for frequency, _ in positions])
    frames = np.concatenate([frame for _, frame in positions])
    order = np.lexsort((frequencies, frames))
    frequencies, frames = frequencies[order], frames[order]
    result: set[Fingerprint] = set()
    base_frame = start_sample // HOP_SIZE
    for index, (frequency, frame) in enumerate(zip(frequencies, frames, strict=True)):
        for other in range(index + 1, min(index + 5, len(frames))):
            delta = int(frames[other] - frame)
            if 0 <= delta <= 200:
                pair = f"{int(frequency)}|{int(frequencies[other])}|{delta}"
                digest = hashlib.sha1(pair.encode("ascii"), usedforsecurity=False).hexdigest()[:20]
                result.add((digest, int(frame) + base_frame))
                if len(result) > MAX_HASHES:
                    raise ValueError("Fingerprint count exceeds the per-window limit")
    return sorted(result, key=lambda item: (item[1], item[0]))


def decode_pcm(data: bytes, encoding: str, channels: int) -> NDArray[np.float32]:
    """
    Convert little-endian PCM to normalized mono audio.

    :param data: Complete interleaved PCM frames.
    :param encoding: pcm_s16le, pcm_s24le, pcm_s32le, pcm_f32le or pcm_f64le.
    :param channels: Number of channels, from 1 to 8.
    """
    formats = {
        "pcm_s16le": ("<i2", 2, 32768),
        "pcm_s32le": ("<i4", 4, 2147483648),
        "pcm_f32le": ("<f4", 4, 1),
        "pcm_f64le": ("<f8", 8, 1),
        "pcm_s24le": ("u1", 3, 8388608),
    }
    if encoding not in formats or not 1 <= channels <= 8:
        raise ValueError("Unsupported PCM encoding or channel count")
    dtype, width, scale = formats[encoding]
    if len(data) % (width * channels):
        raise ValueError("PCM data does not contain complete frames")
    if encoding == "pcm_s24le":
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        integers = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        values = np.where(integers & 0x800000, integers - 0x1000000, integers)
    else:
        values = np.frombuffer(data, dtype=dtype)
    audio = values.astype(np.float32).reshape(-1, channels).mean(axis=1) / scale
    if not np.isfinite(audio).all():
        raise ValueError("PCM contains non-finite samples")
    return np.clip(audio, -1, 1).astype(np.float32)


class WindowSampler:
    """Resample streaming PCM using a bounded rolling analysis buffer."""

    def __init__(self, duration: float, sample_rate: int, channels: int, encoding: str) -> None:
        """
        Initialize a streaming sampler.

        :param duration: Expected track duration in seconds.
        :param sample_rate: Source sample rate, from 8000 to 192000.
        :param channels: Source channel count.
        :param encoding: Source PCM encoding.
        """
        if not 8000 <= sample_rate <= 192000:
            raise ValueError("Unsupported source sample rate")
        decode_pcm(b"", encoding, channels)
        self.windows = plan_windows(duration)
        self.channels = channels
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.position = 0
        self.completed: list[Window] = []
        self._index = 0
        self._parts: list[NDArray[np.float32]] = []
        self._count = 0
        self._buffer_start = 0
        self._finished = False
        self._resampler = (
            soxr.ResampleStream(sample_rate, SAMPLE_RATE, 1, dtype="float32", quality="HQ")
            if sample_rate != SAMPLE_RATE
            else None
        )

    def feed(self, data: bytes, *, last: bool = False) -> list[Fingerprint]:
        """
        Consume PCM and return fingerprints of any newly completed windows.

        :param data: A PCM chunk containing no more than two seconds of source audio.
        :param last: Flush the sampler; subsequent feeds are rejected.
        """
        if self._finished:
            raise ValueError("Sampler has already been finalized")
        audio = decode_pcm(data, self.encoding, self.channels)
        if len(audio) > 2 * self.sample_rate:
            raise ValueError("PCM chunk exceeds the sample-count limit")
        if self._resampler is not None:
            audio = self._resampler.resample_chunk(audio, last=last)
        chunk_end = self.position + len(audio)
        hashes: list[Fingerprint] = []
        total = self.windows[-1].start_sample + self.windows[-1].sample_count
        if chunk_end > total + SAMPLE_RATE:
            raise ValueError("Decoded audio exceeds the declared duration by more than one second")
        if last:
            if abs(chunk_end - total) > SAMPLE_RATE:
                raise ValueError("Decoded audio did not reach the declared track end")
            actual_windows = plan_windows(chunk_end / SAMPLE_RATE)
            if tuple(self.completed) != actual_windows[: len(self.completed)]:
                raise ValueError("Decoded duration is inconsistent with completed coverage")
            self.windows = actual_windows
            total = chunk_end
        if audio.size:
            self._parts.append(audio)
            self._count += len(audio)
        while self._index < len(self.windows):
            window = self.windows[self._index]
            if not last and self._index == len(self.windows) - 1:
                break
            if self._count < window.sample_count:
                break
            samples = np.concatenate(self._parts)
            self._parts.clear()
            extracted = fingerprint(samples[: window.sample_count], window.start_sample)
            next_start = (
                self.windows[self._index + 1].start_sample
                if self._index + 1 < len(self.windows)
                else total
            )
            # Each anchor belongs to one window; the overlap supplies its following peaks.
            hashes.extend(pair for pair in extracted if pair[1] * HOP_SIZE < next_start)
            del extracted
            self.completed.append(window)
            remaining = samples[next_start - self._buffer_start :].copy()
            if remaining.size:
                self._parts.append(remaining)
            self._count = len(remaining)
            self._buffer_start = next_start
            self._index += 1
        self.position = chunk_end
        self._finished = last
        if last:
            self._parts.clear()
        return hashes
