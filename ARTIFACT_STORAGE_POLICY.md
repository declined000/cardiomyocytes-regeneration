# Artifact Storage Policy

This repository keeps **analysis code** in GitHub. **Raw data, intermediate results, figures, and reports** live on your machine and/or external storage (for example Google Drive). They are listed in `.gitignore` and are not committed.

## Keep in GitHub

- Analysis code: `*.py`
- Project metadata: `LICENSE`, this policy, `.gitignore`
- Optional: a small `README` describing how to obtain data and regenerate outputs

## Store Outside GitHub (local or Drive)

Entire directories (all contents):

- `Cardiomyocytes/` — raw video and microscopy
- `final_report/` — figures, CSV exports, `COMBINED_REPORT.txt`, etc.
- `fl_results/`
- `results/`, `results_gcamp/`, `results_v2/`
- `video_inspection/`, `video_results/`
- `fancy_gifs/`

Reproducing paper-style outputs requires placing data in the expected paths and running the scripts locally.

## Before Deleting Local Copies

Only delete local files after all of the following are true:

1. The external copy opens correctly.
2. There is at least one additional backup besides the laptop.
3. You still have a way to regenerate or replace outputs (code + data manifest).
4. A manifest has been recorded with file paths, sizes, and archive location.

## Recommended Cleanup Workflow

1. Upload large raw and generated media to Google Drive.
2. Verify the upload by opening a sample of files and comparing file counts and sizes.
3. If using Google Drive for Desktop, set archived folders to online-only to free local space.
4. Delete local copies only after verification and backup are complete.

## Important Git Note

`.gitignore` only prevents future tracking. It does not remove blobs from **past** commits. After `git rm --cached`, **commit** so new commits no longer contain those paths. **Clones will still download old blobs** until you rewrite history (for example with `git filter-repo`) and force-push, which is optional and disruptive for collaborators.
