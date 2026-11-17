# Library releases and the Dejavu naming transition

Releases are prepared manually, not on every push. The current distribution is
`audio-fingerprint`, version `0.4.1`, with the Python imports `audio_fingerprint`
and `audio_fingerprint.codec`. Python 3.12 or newer is required.

## Version policy

Use `MAJOR.MINOR.PATCH` in `pyproject.toml` and `vMAJOR.MINOR.PATCH` for GitHub
tags. Never move a published tag or replace an existing release's files.

- Patch: compatible fixes that preserve documented interfaces and fingerprint
  output.
- Minor: compatible new capabilities.
- Breaking changes: an explicitly documented compatibility boundary; while
  below 1.0, use a new minor version and make the break prominent in the notes.
  At 1.0 and later, use a new major version.

Package versions, acoustic format IDs, document encoding IDs and server API
versions are different contracts. Do not change the acoustic format or encoded
bytes merely to rename a package. A change to generated hashes or coverage
semantics requires explicit compatibility analysis, fixtures and a format
decision, even if the Python signatures stay the same.

## Prepare a draft release

1. Review the changes, update the version in `pyproject.toml` when needed and
   merge the reviewed source to `main`.
2. In GitHub Actions, manually run **Prepare library release** on `main`.
   Enter the exact package version without `v` and concise release notes.
   Include supported Python versions, breaking changes and upgrade actions
   where applicable.
3. The workflow tests Python 3.12 and 3.14, checks the version/tag, builds a
   source archive and wheel, and smoke-tests the installed wheel outside the
   checkout. It creates a **draft** GitHub release with SHA-256 checksums.
4. Inspect the draft, its target commit, notes and assets. Publish it explicitly
   only when ready. Delete an abandoned draft before retrying the same version;
   never reuse a tag once published.
5. Update consumers to the reviewed release or immutable commit, with the
   corresponding checksum where applicable, and test them before deployment.

This workflow does not publish to PyPI, rename the repository or update any
running installation. Availability of a future PyPI name must be checked
separately. The workflow currently prepares stable numeric versions, including
0.x versions; prerelease identifiers need an explicit workflow extension.

## Planned names: dejavu-fingerprint / dejavu-fingerprint-server

The intended names are `dejavu-fingerprint` for the library and
`dejavu-fingerprint-server` for the service, with corresponding Python modules
`dejavu_fingerprint` and `dejavu_fingerprint_server`. This document is a migration
plan, not an already completed rename. Describe the projects as independent,
Dejavu-style implementations; the names do not establish upstream affiliation
or compatibility with arbitrary Dejavu databases.

Use a staged transition:

1. Check repository and package-index name availability and inventory
   consumers: the fingerprint server, Music Assistant providers, build contexts,
   documentation, release workflows and deployment launch commands.
2. Introduce canonical `dejavu_fingerprint` modules with thin, tested
   compatibility re-exports for `audio_fingerprint` and
   `audio_fingerprint.codec`. Move the implementation once, not into duplicate
   copies. Preserve signatures, symbols and fixture output.
3. Change the distribution metadata and document that old/new distribution
   names must not be installed as competing owners of the same files. In
   existing environments, remove the old distribution before installing the
   new one; build clean container images. Retain the old public import paths
   during a stated deprecation window.
4. Rename GitHub repositories only after reviewing the dependency update plan.
   Replace source URLs and release links in consumers explicitly; do not rely
   solely on repository redirects. Keep published old releases accessible.
5. Release the compatible library first. Update and validate the fingerprint
   server and MA consumers against it before renaming the server distribution,
   module and launch commands.
6. Preserve `/api/v1/dejavu/*`, existing acoustic/encoding identifiers,
   submission identities, database schemas and stored service state unless a
   separate, approved migration actually requires a change.

The earlier `music_assistant_fingerprint` import was already removed in 0.4.0;
do not imply it is supported by this plan. A server release workflow follows
after library release and naming conventions have been exercised.
