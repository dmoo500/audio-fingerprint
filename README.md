# Audio Fingerprint Library

An experimental shared landmark-fingerprint library for Music Assistant analysis providers
and an independently hosted recognition service. It has no database drivers, microphone
hardware dependency, audio-file downloads, or file-writing code.
This is an independent prototype, not an official Music Assistant or OHF project.
It is MIT licensed; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Format

`ma-dejavu-landmarks-v2` uses Dejavu-style frequency-pair hashes: 44,100 Hz mono audio,
4,096-sample Hann FFT windows, a 2,048-sample hop, square peak neighborhoods of radius 10,
a 10 dB amplitude threshold with signed-16-bit scaling, fan-out 5, and SHA-1 truncated
to 20 hexadecimal characters. Offsets are integer FFT-hop positions relative to the track.

The algorithm is a new implementation of the landmark approach, not the original Dejavu
package. Compatibility with an existing Dejavu database must be verified against its
actual implementation and parameters; matching hash shapes alone does not establish it.

## Coverage and limits

Reference analysis covers the entire decoded track using windows of at most 30 seconds,
with a hop-aligned stride of approximately 20 seconds. The approximately ten-second overlap
supplies context for frequency-pair hashes crossing section boundaries; each hash anchor
belongs to one window, preventing duplicate hash/offset pairs. Tracks longer than 30 minutes
are rejected. Audio is resampled continuously with soxr HQ; the rolling mono PCM buffer holds
at most one window plus a bounded input chunk, not the whole recording.

The final window is finalized only at end of input. Declared duration is allowed to differ
from decoded duration by at most one second for rounded metadata; the actual samples determine
coverage, with no padding or discarded tail. Larger mismatches fail explicitly. A caller
must record the actual decoded duration and must not publish cancelled analyses.
Coverage means PCM was processed, not that silence, noise or every microphone capture will
produce a match. Reference and query audio use the same normalization and landmark parameters.

Per-window extraction is capped at 100,000 landmarks. The provider and server share a
1,000,000-landmark / 36,000,000-byte document limit per track. Extraction and serialization
must run off the event loop with the host's resource caps; full coverage necessarily consumes
more work than sparse sampling.

Version 0.2.1 bounds FFT intermediates to 64 frames per batch and peak-filter
intermediates to the same batch plus a ten-frame halo. Float64 precision, frame
alignment, full-window peak neighborhoods and hash ordering are unchanged.
The power array is converted to decibels in place, and consumed rolling-buffer
chunks are released before extraction. Compatibility tests compare the exact
hashes and offsets with the original unbatched algorithm, including silence,
tones, noisy audio and batch boundaries. Coverage and document formats remain
unchanged; upgrading from 0.2.0 requires neither reanalysis nor a data migration.

Version 0.2.2 also avoids retaining the entire window's spectrogram: each FFT
batch includes the ten-frame peak-filter halo and is consumed immediately.
Only the peaks survive between batches. This trades repeated halo FFTs for
bounded spectral memory; exact full-track hash/offset compatibility is retained.

Legacy `ma-dejavu-landmarks-v1` references retain their three-window sampled coverage.
Their landmark algorithm is unchanged and remains searchable, but new full-track documents
use v2 because their coverage and limits differ. Update receivers before enabling v2 producers.

## Development

```sh
python -m pip install -e .
python -m pytest
```

This package is not published on PyPI. Integrations use an exact Git commit
from this repository, or install it locally. Do not claim compatibility with
an existing Dejavu database without testing that database and its parameters.
