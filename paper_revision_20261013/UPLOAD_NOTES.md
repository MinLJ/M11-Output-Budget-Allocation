# M11 manuscript revision package — upload notes

Uploaded 2026-10-14 on branch `upload/m11-paper-revision`, copied from the
frozen delivery package dated 2026-10-13.  The files below are byte-identical
copies of the package contents, with the following differences:

- The docx and PDF renders (`M11_manuscript_v2_revised.docx/.pdf`,
  `Response_to_Comments.docx/.pdf`) are **not included**.  They are generated
  from the Markdown sources with pandoc (docx, native OMML equations) and
  LibreOffice (PDF); the Markdown files are the single source of truth.
- The four manuscript figures (`01_manuscript/figures/fig1–fig4.png`, 6.7 MB
  total, 400 dpi crops from the original manuscript PDF) are **not included**,
  per this repository's policy that large assets stay outside ordinary Git
  (`.gitignore` covers `*.png`).  Registered here by identity only: fig1.png
  1.9 MB, fig2.png 3.5 MB, fig3.png 1.6 MB, fig4.png 0.05 MB.  As a result
  the image links in `01_manuscript/M11_manuscript_v2_revised.md` do not
  resolve in this repository; the figures are available in the frozen
  delivery package (`M11_paper_revision_20261013.zip`).
- `03_reproducibility/recompute_seed_spread.py` contains a machine-local
  `ROOT` constant.  Four of its five per-seed sources live in this repository
  under `experiments/`; the fifth is the Faster R-CNN arm's external member
  delivery (`member_faster_rcnn_delivery/results/main_results.csv`).  Adjust
  `ROOT` before running.  Nothing was retrained or re-evaluated for this
  package; the script only re-aggregates frozen per-seed result CSVs.

| Path | Contents |
|---|---|
| `01_manuscript/` | Revised manuscript (Markdown) |
| `02_response/` | Point-by-point response to the supervisor's review comments |
| `03_reproducibility/` | Seed-spread recomputation script and its report/CSV |
| `README_交付说明.md` | Chinese delivery note for the frozen package (manifest, per-comment mapping, open items) |

Status: revised per supervisor comments; **awaiting supervisor review before
any submission**.  No models were retrained, no allocation was re-run and no
metric was recomputed for this upload.
