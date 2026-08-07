# Upload guide for the existing ANCHOR repository

The release package is a replacement working tree and intentionally omits
`.git/`. Preserve the existing Git history and remote.

## Safe procedure

1. Back up `D:\github_repository\ANCHOR` or create a fresh clone.
2. Inspect and preserve any current work in the existing repository:

   ```bash
   git status
   git diff
   git add --all
   git commit -m "Preserve pre-release repository updates"
   ```

3. Create a release branch:

   ```bash
   git switch -c reproducibility-v1.2.0
   ```

4. Replace the branch working-tree contents with the contents of
   `ANCHOR_GitHub_release_v1.2.0/`, preserving `.git/`. Copy the package
   contents, not the outer directory as a nested folder.
5. Validate the copied repository:

   ```bash
   python scripts/maintenance/validate_release.py
   git status
   ```

6. If anything changed after copying, rebuild checksums and validate again:

   ```bash
   python scripts/maintenance/build_manifest.py
   python scripts/maintenance/validate_release.py
   ```

7. Commit, push, and inspect the rendered README:

   ```bash
   git add --all
   git commit -m "Release reproducibility package v1.2.0"
   git push -u origin reproducibility-v1.2.0
   ```

8. Merge after verification, create annotated tag `v1.2.0`, and create the
   GitHub release described in `VERSIONED_RELEASE.md`.

## Author decisions before publication

- Add an approved software/data license.
- Insert the final manuscript citation.
- Insert the raw-read archive accession when available.
- Add the GitHub release URL and optional Zenodo/Figshare DOI to the manuscript
  and response letter.
