# Versioned GitHub release procedure

## Release identifier

Use the annotated Git tag and GitHub release `v1.2.0`.

```bash
git add .
git commit -m "Release complete ANCHOR workflow v1.2.0"
git tag -a v1.2.0 -m "ANCHOR workflow release v1.2.0"
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

> Version 1.2.0 provides processed datasets; a manuscript-aligned fixed 80:20
> holdout with 10-fold cross-validation restricted to the 80% development
> set; ten-seed histories and predictions; calibration parameters; bootstrap
> confidence intervals; uncertainty and residual analyses; and environment
> locks. Only the classifier and regressor copied byte-for-byte from the
> pTyr_antibody-analog inference archive are distributed as SavedModels.
> Evaluation-generated SavedModels and intermediate seed weights are omitted.
> Test sets are not used for early stopping, calibration, threshold selection,
> or model selection.
