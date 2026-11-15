"""Lossless, bounded landmark encoding independent of the acoustic fingerprint format."""

from __future__ import annotations

import base64
import binascii
from typing import Any

from audio_fingerprint import HOP_SIZE, MAX_DURATION, MAX_TRACK_HASHES, SAMPLE_RATE

ENCODING_ID = "sha1-80-offset-delta-v1"
MAX_OFFSET = MAX_DURATION * SAMPLE_RATE // HOP_SIZE
MAX_PACKED_BYTES = MAX_TRACK_HASHES * 13
MAX_ENCODED_BYTES = 4 * ((MAX_PACKED_BYTES + 2) // 3)


def encode_document(document: dict[str, Any]) -> dict[str, Any]:
    """
    Encode a canonical landmark document without changing acoustic data.

    :param document: Legacy document with unique fingerprints sorted by hash and offset.
    :raises ValueError: If fingerprints or their ordering are invalid.
    """
    if "fingerprints_encoding" in document:
        decode_document(document)
        return document.copy()
    pairs = document.get("fingerprints")
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= MAX_TRACK_HASHES:
        raise ValueError("Invalid fingerprint count")
    validated: list[tuple[str, int]] = []
    previous: tuple[str, int] | None = None
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Invalid fingerprint pair")
        digest, offset = pair
        if (
            not isinstance(digest, str)
            or len(digest) != 20
            or any(char not in "0123456789abcdef" for char in digest)
            or type(offset) is not int
            or not 0 <= offset <= MAX_OFFSET
        ):
            raise ValueError("Invalid fingerprint hash or offset")
        value = (digest, offset)
        if previous is not None and value <= previous:
            raise ValueError("Fingerprints must be unique and in canonical order")
        validated.append(value)
        previous = value
    payload = bytearray()
    offset_before = 0
    for digest, offset in sorted(validated, key=lambda pair: (pair[1], pair[0])):
        payload.extend(bytes.fromhex(digest))
        delta = offset - offset_before
        while delta >= 128:
            payload.append((delta & 127) | 128)
            delta >>= 7
        payload.append(delta)
        offset_before = offset
    result = document.copy()
    result["fingerprints_encoding"] = ENCODING_ID
    result["fingerprints"] = base64.b64encode(payload).decode("ascii")
    return result


def decode_document(document: dict[str, Any]) -> dict[str, Any]:
    """
    Decode a legacy or compact document into its original canonical JSON representation.

    :param document: Document with legacy pairs or the supported compact encoding.
    :raises ValueError: If the encoding, record limits or ordering are invalid.
    """
    if "fingerprints_encoding" not in document:
        return document.copy()
    if document["fingerprints_encoding"] != ENCODING_ID:
        raise ValueError("Unsupported fingerprint encoding")
    encoded = document.get("fingerprints")
    if not isinstance(encoded, str) or len(encoded) > MAX_ENCODED_BYTES:
        raise ValueError("Encoded fingerprints exceed the size limit")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as err:
        raise ValueError("Invalid fingerprint base64") from err
    if not payload or len(payload) > MAX_PACKED_BYTES:
        raise ValueError("Packed fingerprints exceed the size limit")
    pairs: list[tuple[str, int]] = []
    position = 0
    previous: tuple[int, str] | None = None
    offset = 0
    while position < len(payload):
        if len(pairs) >= MAX_TRACK_HASHES or position + 11 > len(payload):
            raise ValueError("Truncated or excessive fingerprints")
        digest = payload[position : position + 10].hex()
        position += 10
        delta = 0
        for shift in (0, 7, 14):
            if position >= len(payload):
                raise ValueError("Truncated fingerprint offset")
            byte = payload[position]
            position += 1
            delta |= (byte & 127) << shift
            if not byte & 128:
                if shift and byte == 0:
                    raise ValueError("Noncanonical fingerprint offset")
                break
        else:
            raise ValueError("Fingerprint offset overflow")
        offset += delta
        current = (offset, digest)
        if offset > MAX_OFFSET or (previous is not None and current <= previous):
            raise ValueError("Invalid fingerprint offset or ordering")
        pairs.append((digest, offset))
        previous = current
    result = document.copy()
    result.pop("fingerprints_encoding")
    result["fingerprints"] = [list(pair) for pair in sorted(pairs)]
    return result
