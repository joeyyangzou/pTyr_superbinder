# Versioned GitHub release procedure

## Release identifier

Use the annotated Git tag and GitHub release `v1.2.0`.

```bash
git add .
git commit -m "Release reproducibility package v1.2.0"
git tag -a v1.2.0 -m "ANCHOR reproducibility release v1.2.0"
git push origin main
git push origin v1.2.0
```

Create a GitHub release from the tag and attach the release ZIP. If possible,
archive the tagged release with Zenodo and report the DOI in the manuscript.

## Before publishing

1. Run `python scripts/maintenance/validate_release.py`.
2. Regenerate `MANIFEST.tsv` and `CHECKSUMS.sha256` after the final edit.
3. Confirm that no identifiers, tokens, passwords, private paths, or unrelated
   unpublished data are present.
4. Confirm that every file is below GitHub's 100 MB limit.
5. Add a software/data license approved by the authors and institution.
6. Add the final manuscript citation, raw-read accession, and archival DOI when
   available.
7. Keep supplied reference results immutable and write reruns to a new output
   directory.

## Suggested release notes

> Version 1.2.0 provides processed datasets, frozen random and
> Hamming-distance-separated partitions, ten-seed histories and predictions,
> calibration parameters, bootstrap confidence intervals, uncertainty and
> residual analyses, environment locks, and the latest validation-selected
> classification and regression SavedModels. Historical models and
> intermediate seed weights are not distributed. Test sets are not used for
> early stopping, calibration, threshold selection, or model selection.
