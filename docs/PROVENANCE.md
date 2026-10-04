# Release provenance

## Assembly policy

The repository was assembled from copies of the research source tree. Original
files were not edited. `SOURCE_MANIFEST.tsv` records the original repository-
relative path and both the original and public-copy SHA-256 checksums for every
copied file.

## Changes made to the public copies

The copied files retain their original scientific implementation. Public-release
changes are limited to:

1. Replacing personal absolute paths in usage examples with repository-relative
   paths.
2. Making the two shell launchers resolve their own directory rather than a
   specific workstation path.
3. Replacing a private CRNet provenance path with the public upstream URL.
4. Adding root-level setup, input-layout, licensing, mapping, and validation
   documentation.
5. Adding the effective-rank helper used by several evaluation scripts from the
   MIT-licensed companion repository.
6. Loading SciPy only when the effective-rank Wasserstein metric is requested,
   so the NumPy-only helpers remain independently importable.

No metric definition, model architecture, threshold, seed, checkpoint grid, or
reported experimental setting was intentionally changed.

## Upstream components

- The DDIM training lineage is associated with the MIT-licensed Telefónica
  Scientific Research `GenAI_Channel_Modeling` release.
- The CSI-compression model adapts CRNet by Kylin Lu and contributors (MIT).
- The beam-alignment model adapts DL-GF by Yuqiang Heng and contributors
  (GPL-3.0).

See `THIRD_PARTY_NOTICES.md` and `LICENSES/` for the corresponding notices.
