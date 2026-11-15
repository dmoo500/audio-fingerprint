"""Compare exact-compatible extraction and lossless landmark representations locally."""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import struct
import time
import zlib
from collections.abc import Callable

import numpy as np

from music_assistant_fingerprint import SAMPLE_RATE, Fingerprint, WindowSampler


def measure[T](call: Callable[[], T], iterations: int) -> tuple[T, float]:
    """Return the last result and median wall time in milliseconds."""
    result = call()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        result = call()
        samples.append((time.perf_counter() - started) * 1000)
    return result, statistics.median(samples)


def extract(duration: int, kind: str) -> list[Fingerprint]:
    """Generate a bounded deterministic fixture without reading or writing audio files."""
    sampler = WindowSampler(duration, SAMPLE_RATE, 1, "pcm_f32le")
    rng = np.random.default_rng(20261115)
    result = []
    for second in range(duration):
        audio = rng.normal(0, 0.1, SAMPLE_RATE).astype(np.float32)
        if kind == "tone":
            positions = np.arange(SAMPLE_RATE) + second * SAMPLE_RATE
            audio = (0.5 * np.sin(positions * 2 * np.pi * 440 / SAMPLE_RATE)).astype(np.float32)
        result.extend(sampler.feed(audio.tobytes()))
    result.extend(sampler.feed(b"", last=True))
    return sorted(set(result))


def pack(pairs: list[Fingerprint], delta: bool) -> bytes:
    """Encode the unchanged ten-byte hashes with fixed-width or delta offsets."""
    output = bytearray()
    previous = 0
    ordered = sorted(pairs, key=lambda pair: (pair[1], pair[0])) if delta else pairs
    for digest, offset in ordered:
        output.extend(bytes.fromhex(digest))
        if delta:
            value = offset - previous
            if value < 0:
                raise ValueError("Delta encoding requires nondecreasing offsets")
            while value >= 128:
                output.append((value & 127) | 128)
                value >>= 7
            output.append(value)
            previous = offset
        else:
            output.extend(struct.pack(">I", offset))
    return bytes(output)


def unpack(data: bytes, delta: bool) -> list[Fingerprint]:
    """Decode benchmark records and reject truncated or overflowing offsets."""
    result = []
    position = 0
    previous = 0
    while position < len(data):
        if position + 10 > len(data):
            raise ValueError("Truncated hash")
        digest = data[position : position + 10].hex()
        position += 10
        if delta:
            value = 0
            for shift in range(0, 35, 7):
                if position == len(data):
                    raise ValueError("Truncated delta")
                byte = data[position]
                position += 1
                value |= (byte & 127) << shift
                if not byte & 128:
                    break
            else:
                raise ValueError("Delta exceeds five bytes")
            offset = previous + value
            if offset > 0xFFFFFFFF:
                raise ValueError("Offset exceeds uint32")
            previous = offset
        else:
            if position + 4 > len(data):
                raise ValueError("Truncated offset")
            offset = struct.unpack_from(">I", data, position)[0]
            position += 4
        result.append((digest, offset))
    return sorted(result)


def compare(pairs: list[Fingerprint], iterations: int) -> list[dict[str, str | int | float]]:
    """Measure full encoding and decoding, checking every reconstructed hash and offset."""

    def json_encode() -> bytes:
        return json.dumps(pairs, separators=(",", ":")).encode()

    def json_decode(data: bytes) -> list[Fingerprint]:
        return [(digest, offset) for digest, offset in json.loads(data)]

    representations: list[tuple[str, Callable[[], bytes], Callable[[bytes], list[Fingerprint]]]] = [
        ("json", json_encode, json_decode),
        ("binary_u32", lambda: pack(pairs, False), lambda data: unpack(data, False)),
        ("binary_delta", lambda: pack(pairs, True), lambda data: unpack(data, True)),
    ]
    baseline = len(json_encode())
    results = []
    for name, encode, decode in representations:
        for level in (0, 1, 6, 9):

            def encode_payload(encode: Callable[[], bytes] = encode, level: int = level) -> bytes:
                raw = encode()
                return zlib.compress(raw, level) if level else raw

            payload, encode_ms = measure(encode_payload, iterations)

            def decode_payload(
                payload: bytes = payload,
                level: int = level,
                decode: Callable[[bytes], list[Fingerprint]] = decode,
            ) -> list[Fingerprint]:
                raw = zlib.decompress(payload) if level else payload
                return decode(raw)

            restored, decode_ms = measure(decode_payload, iterations)
            if restored != pairs:
                raise RuntimeError(f"Round-trip mismatch: {name}, level {level}")
            results.append(
                {
                    "representation": name,
                    "zlib_level": level,
                    "bytes": len(payload),
                    "base64_bytes": len(base64.b64encode(payload)),
                    "reduction_percent": round(100 * (1 - len(payload) / baseline), 2),
                    "encode_median_ms": round(encode_ms, 3),
                    "decode_median_ms": round(decode_ms, 3),
                }
            )
    return results


def main() -> None:
    """Print measurements, not a proposed replacement for the existing wire format."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration", type=int, default=60, choices=range(1, 1801), metavar="1-1800"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, choices=range(1, 101), metavar="1-100"
    )
    parser.add_argument("--kind", choices=("noise", "tone"), default="noise")
    args = parser.parse_args()
    pairs, extraction_ms = measure(
        lambda: extract(args.duration, args.kind), min(args.iterations, 5)
    )
    print(
        json.dumps(
            {
                "fixture": f"synthetic_{args.kind}",
                "duration_seconds": args.duration,
                "landmarks": len(pairs),
                "extraction_median_ms": round(extraction_ms, 3),
                "iterations": args.iterations,
                "methodology": "Complete encode/decode costs; "
                "no metadata, base64 time or networking. "
                "Synthetic audio is not a representative music collection.",
                "representations": compare(pairs, args.iterations),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
