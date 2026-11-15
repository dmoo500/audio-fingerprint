# Audio Fingerprint Library

An experimental shared landmark-fingerprint library for Music Assistant analysis providers and an independently hosted recognition service. It has no database drivers, microphone hardware dependency, audio-file downloads, or file-writing code. This is an independent prototype, not an official Music Assistant or OHF project. It is MIT licensed; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Format

`ma-dejavu-landmarks-v2` uses Dejavu-style frequency-pair hashes: 44,100 Hz mono audio, 4,096-sample Hann FFT windows, a 2,048-sample hop, square peak neighborhoods of radius 10, a 10 dB amplitude threshold with signed-16-bit scaling, fan-out 5, and SHA-1 truncated to 20 hexadecimal characters. Offsets are integer FFT-hop positions relative to the track.

The algorithm is a new implementation of the landmark approach, not the original Dejavu package. Compatibility with an existing Dejavu database must be verified against its actual implementation and parameters; matching hash shapes alone does not establish it.

## Coverage and limits

Reference analysis covers the entire decoded track using windows of at most 30 seconds, with a hop-aligned stride of approximately 20 seconds. The approximately ten-second overlap supplies context for frequency-pair hashes crossing section boundaries; each hash anchor belongs to one window, preventing duplicate hash/offset pairs. Tracks longer than 30 minutes are rejected. Audio is resampled continuously with soxr HQ; the rolling mono PCM buffer holds at most one window plus a bounded input chunk, not the whole recording.

The final window is finalized only at end of input. Declared duration is allowed to differ from decoded duration by at most one second for rounded metadata; the actual samples determine coverage, with no padding or discarded tail. Larger mismatches fail explicitly. A caller must record the actual decoded duration and must not publish cancelled analyses. Coverage means PCM was processed, not that silence, noise or every microphone capture will produce a match. Reference and query audio use the same normalization and landmark parameters.

Per-window extraction is capped at 100,000 landmarks. The provider and server share a 1,000,000-landmark / 36,000,000-byte document limit per track. Extraction and serialization must run off the event loop with the host's resource caps; full coverage necessarily consumes more work than sparse sampling.

Version 0.2.1 bounds FFT intermediates to 64 frames per batch and peak-filter intermediates to the same batch plus a ten-frame halo. Float64 precision, frame alignment, full-window peak neighborhoods and hash ordering are unchanged. The power array is converted to decibels in place, and consumed rolling-buffer chunks are released before extraction. Compatibility tests compare the exact hashes and offsets with the original unbatched algorithm, including silence, tones, noisy audio and batch boundaries. Coverage and document formats remain unchanged; upgrading from 0.2.0 requires neither reanalysis nor a data migration.

Version 0.2.2 also avoids retaining the entire window's spectrogram: each FFT batch includes the ten-frame peak-filter halo and is consumed immediately. Only the peaks survive between batches. This trades repeated halo FFTs for bounded spectral memory; exact full-track hash/offset compatibility is retained.

Legacy `ma-dejavu-landmarks-v1` references retain their three-window sampled coverage. Their landmark algorithm is unchanged and remains searchable, but new full-track documents use v2 because their coverage and limits differ. Update receivers before enabling v2 producers.

## Development

```sh
python -m pip install -e .
python -m pytest
```

### Lossless optimization benchmarks

Earlier shipped memory optimizations bound FFT work to 64-frame batches with a peak-filter halo and avoid retaining a full-window spectrogram. The streaming sampler retains a bounded rolling PCM window. Exact compatibility tests cover those changes, but no measured peak-memory reduction is claimed.

The extraction loop converts peak coordinates to Python integers once before pairing, avoiding repeated NumPy scalar conversion without changing hashes, offsets, FFT precision or bounded batch sizes. Exact compatibility is checked against the original algorithm. An alternating local comparison on 30-second synthetic fixtures measured about 2-4% less extraction time for noise and silence-gap inputs, with no meaningful change for a tone. These are machine-specific measurements, not a performance guarantee.

| 30-second fixture | Before | After | Change |
| --- | ---: | ---: | --- |
| Noise | 69.662 ms | 67.110 ms | About 3.66% less extraction time |
| Silence gaps | 58.322 ms | 57.122 ms | About 2.06% less extraction time |
| Tone | 46.348 ms | 46.312 ms | No meaningful improvement |

These historical measurements used an alternating local comparison with 30 runs. The scalar-conversion optimization is shipped in library 0.3.0; the measurements do not imply a similar improvement for every track.

Compare canonical JSON, ten-byte hashes with fixed-width offsets, and ten-byte hashes with delta-varint offsets, optionally compressed with zlib:

```sh
PYTHONPATH=src python scripts/benchmark_compression.py --duration 60 --iterations 10
PYTHONPATH=src python scripts/benchmark_compression.py --kind tone --duration 60
```

The benchmark verifies exact hash/offset round trips and measures serialization, compression, decompression and parsing. Delta encoding includes sorting by offset; decoding restores the canonical hash/offset order. It reports raw payload and base64 sizes separately. Metadata, base64 processing and network costs are not measured. The deterministic noise/tone fixtures are not a representative music collection.

### Compact documents (0.3.0)

`music_assistant_fingerprint.codec.encode_document` converts canonical, unique, hash/offset-sorted pairs into a compact document. `decode_document` accepts compact documents and legacy JSON and restores the original canonical pairs. Other fields, acoustic hashes and offsets are unchanged; this does not require reanalysis.

Compact documents set `fingerprints_encoding` to `sha1-80-offset-delta-v1` and store `fingerprints` as base64. Records are ordered by offset, then hash: ten binary hash bytes followed by the minimal unsigned base-128 varint delta from the previous offset (initially zero). The decoder bounds encoded size, record count and offsets, rejects malformed or duplicate records and restores hash-first ordering. No zlib decompression is used. Additional compression saved little in the experiments relative to its CPU cost. On 60-second synthetic noise and tone fixtures, the production landmark JSON section was about 49-50% smaller including base64 and the encoding field. For 23,986 noise landmarks, a local median measured about 42 ms encoding and 21 ms decoding; compact storage trades this additional work for reduced storage and upload size. Actual music and machines will differ.

| Representation | Approximate size saving against canonical JSON pairs | Status |
| --- | ---: | --- |
| JSON with zlib | 49% | Experiment |
| Ten-byte hashes with fixed-width offsets, raw binary | 53% | Experiment |
| Ten-byte hashes with delta-varint offsets, raw binary | 63% | Basis of shipped codec; raw size excludes base64 |
| Delta-varint binary with zlib | 64% | Experiment; little additional saving |
| Production compact landmark JSON section | 49-50% | Shipped in 0.3.0; includes base64 and encoding field |

The earlier raw representation experiments used synthetic v2 documents and three existing processed v1 reference documents. The production JSON measurement used 60-second synthetic noise/tone fixtures. These are different datasets and size boundaries, not interchangeable percentages. Production encoding and decoding of 23,986 noise landmarks took approximately 42 and 21 ms median respectively. Compact encoding reduces document size, not landmark count or acoustic precision.

The [server results overview](https://github.com/dmoo500/audio-fingerprint-server#benchmark-results-and-shipped-optimizations) includes charts for database migration and short-capture recognition, with exact tables and limitations. Recognition rates belong to the server's indexed collection and acceptance rules, not to this extraction library alone. The short-capture tests used synthetic audio, not real microphone recordings, and did not change thresholds.

Submission identity is still SHA-256 of **decoded canonical JSON**, without the encoding field, using sorted keys, compact separators and Python's default ASCII escaping. Deploy a codec-aware receiver and gateway before enabling compact producers; discover support through the receiver's authenticated capabilities endpoint. Existing legacy JSON uploads remain supported. Migrations must retain noncanonical or invalid legacy records instead of silently changing their identity.

The fixed-width and zlib variants remain benchmark experiments, not supported encodings. Acoustic v1/v2 format IDs do not change for lossless compression. Cryptographic hash bytes themselves are mostly incompressible; binary packing removes hexadecimal overhead, while delta encoding primarily reduces the offsets.

This package is not published on PyPI. Integrations use an exact Git commit from this repository, or install it locally. Do not claim compatibility with an existing Dejavu database without testing that database and its parameters.
