# Packaging note

The engineering and development runs loaded the pinned files through Python
namespace packages. After those runs, two inert package initializers were
added so that embedding this folder in the full BPD repository always selects
the bundled source snapshot. They contain docstrings only. The four upstream
computational files, task code and tests retain their original hashes.

The independent replay and five tests pass with the packaged import layout.
`manifest_before_run.json` records each historical run's source set.
`PROVENANCE.json` pins the upstream files; the packaging files were added after
data collection and are excluded from those historical manifests.
