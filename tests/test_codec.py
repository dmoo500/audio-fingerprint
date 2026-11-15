"""Compact documents preserve identities and reject malformed or excessive records."""

import base64
import json

import pytest

import audio_fingerprint.codec as codec
from audio_fingerprint.codec import (
    ENCODING_ID,
    MAX_OFFSET,
    decode_document,
    encode_document,
)


def document() -> dict:
    return {
        "format_id": "ma-dejavu-landmarks-v2",
        "fingerprints": [["0" * 20, MAX_OFFSET], ["a" * 20, 0], ["f" * 20, 128]],
        "metadata": {"title": "untouched"},
    }


def test_round_trip_preserves_canonical_document_and_input() -> None:
    original = document()
    saved = json.dumps(original, sort_keys=True)
    compact = encode_document(original)
    assert compact["fingerprints_encoding"] == ENCODING_ID
    assert decode_document(compact) == original
    assert encode_document(compact) == compact
    assert json.dumps(original, sort_keys=True) == saved
    assert decode_document(original) == original


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"a" * 10,
        b"a" * 10 + b"\x80",
        b"a" * 10 + b"\x80\x00",
        b"a" * 10 + b"\xff\xff\xff",
        b"a" * 10 + b"\xff\xff\x7f",
        (b"a" * 10 + b"\x00") * 2,
        b"b" * 10 + b"\x00" + b"a" * 10 + b"\x00",
    ],
)
def test_malformed_packed_records_rejected(payload: bytes) -> None:
    compact = {
        "fingerprints_encoding": ENCODING_ID,
        "fingerprints": base64.b64encode(payload).decode(),
    }
    with pytest.raises(ValueError):
        decode_document(compact)


@pytest.mark.parametrize("encoded", ["!", "ä", "AAAA==="])
def test_invalid_base64_rejected(encoded: str) -> None:
    with pytest.raises(ValueError):
        decode_document({"fingerprints_encoding": ENCODING_ID, "fingerprints": encoded})


def test_unknown_encoding_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        decode_document({"fingerprints_encoding": "future", "fingerprints": ""})


@pytest.mark.parametrize(
    "pairs",
    [
        [],
        [["F" * 20, 0]],
        [["a" * 20, True]],
        [["a" * 20, -1]],
        [["a" * 20, MAX_OFFSET + 1]],
        [["b" * 20, 0], ["a" * 20, 0]],
        [["a" * 20, 0], ["a" * 20, 0]],
    ],
)
def test_invalid_legacy_pairs_rejected(pairs: list) -> None:
    with pytest.raises(ValueError):
        encode_document({"fingerprints": pairs})


def test_landmark_count_cap_applies_to_both_encodings(monkeypatch: pytest.MonkeyPatch) -> None:
    original = {"fingerprints": [[f"{index:020x}", 0] for index in range(4)]}
    compact = encode_document(original)
    monkeypatch.setattr(codec, "MAX_TRACK_HASHES", 3)
    with pytest.raises(ValueError, match="count"):
        encode_document(original)
    with pytest.raises(ValueError, match="excessive"):
        decode_document(compact)


def test_encoded_and_packed_size_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    compact = encode_document(document())
    monkeypatch.setattr(codec, "MAX_ENCODED_BYTES", len(compact["fingerprints"]) - 1)
    with pytest.raises(ValueError, match="Encoded"):
        decode_document(compact)
    monkeypatch.setattr(codec, "MAX_ENCODED_BYTES", len(compact["fingerprints"]))
    monkeypatch.setattr(codec, "MAX_PACKED_BYTES", 10)
    with pytest.raises(ValueError, match="Packed"):
        decode_document(compact)
