# JAS submission checklist

Target journal: *The Journal of the Astronautical Sciences* (Springer). Build the PDF with
`pdflatex main && bibtex main && pdflatex main && pdflatex main` in `paper/`. The build needs
`sn-jnl.cls` and `sn-mathphys-num.bst`, which are in this folder, and `../figures/jas/`.

## Done in the manuscript

- [x] Springer Nature template (`sn-jnl`), numbered references in square brackets (`sn-mathphys-num`).
- [x] Title page: author, department, institute, city, postcode, country, e-mail, ORCID.
- [x] Abstract of about 245 words (limit 150–250). Abbreviations are spelled out.
- [x] Six keywords.
- [x] At most three heading levels, all numbered.
- [x] Declarations: funding, competing interests, data and code availability, author
      contributions, ethics.
- [x] References have DOIs where one exists. They were checked against Crossref and publisher
      pages. Jo et al. is updated to the 2026 *Astrodynamics* paper, and Iannamorelli & LeGrand to
      JAS 72(1) 2025.
- [x] Figures are `figures/jas/Fig1…Fig13`, in vector PDF and 600 dpi TIFF:
  - 84 or 174 mm wide, with Arial/Helvetica-type lettering of 8–12 pt;
  - no titles inside the figures; panels are labelled **a**, **b**, …;
  - orbits and filters are told apart by line style or marker as well as colour;
  - captions start with "Fig." and have no final full stop.
- [x] The numbers in the text and tables are checked by `scripts/verify_claims.py`, which should
      show 0 FAIL and 0 TEXT.

## You must do before submitting

- [ ] **Write the AI-use declaration.** JAS requires it in the Methods section. The comment
      `%% [Declaration of AI-assisted tools ...]` at the end of Section 3 marks where it goes.
- [ ] **Funding.** Replace "did not receive support" if a fellowship or grant paid for your time.
- [ ] **Acknowledgements.** Currently "None"; name any supervisor or colleague who helped.
- [ ] **Code availability.**
  - [ ] Make the GitHub repository public.
  - [ ] Create the Zenodo release.
  - [ ] Replace `zenodo.XXXXXXX` in the Data and code availability statement.
- [ ] **Institutional approval.** JAS treats submission as tacit approval by your institute.
      Tell your supervisor or department before you submit.
- [ ] **Exclusivity.** Do not submit this paper anywhere else while JAS is reviewing it. An arXiv
      preprint is normally allowed; if you post one, say so in the cover letter.
- [ ] **Suggested reviewers (optional).** Choose people with no involvement in this work, and
      give their institutional e-mails. Anyone who reads your draft or endorses you on arXiv no
      longer qualifies.
- [ ] **Cover letter.** Give the one-paragraph contribution, why the paper fits JAS, and a
      statement that it is not under consideration elsewhere.
- [ ] **Editorial Manager.** Upload `main.tex`, `references.bib`, `sn-jnl.cls`,
      `sn-mathphys-num.bst`, `Fig1–Fig13` (PDF or TIFF) and the compiled `main.pdf`.

## Evidence that is right but statistically thin (`verify_claims.py` shows WEAK)

| Claim | Why it is thin | Where it is disclosed |
| :-- | :-- | :-- |
| UT predictor log-R² = 0.90 at N = 100 | bootstrap 95 % CI 0.74–0.98 | Sec. 8 gives the interval |
| Linear predictor log-R² = 0.93 at N = 10 | bootstrap 95 % CI 0.81–0.99 | not given in the text; fine as supporting evidence only |
| Ephemeris UT log-R² = 0.68 at N = 100 | only 28 finite cases, CI 0.25–0.99 | Sec. 9 gives n and the interval |
| Ephemeris, L1 halo, 1-day arcs | only 4 cases | Table 5 column n |

## Numbers that changed after re-verification (draft v1 → this version)

- Reacquisition, L1 halo, re-run with 100 runs instead of 10. The 10-run values were noise.

  | Filter | Consistent, v1 (10 runs) | Consistent, now (100 runs) |
  | :-- | :-: | :-: |
  | EKF | 10 % | 25 % |
  | UKF | 70 % | 55 % |
  | GM-UKF | 100 % | 99 % |
  | PF→UKF | 70 % | 91 % |

  With 10 runs, GM-UKF vs UKF was not significant (p = 0.21); with 100 runs, p < 10⁻¹⁴.
- Linear false custody: "after about one week" → after about 10 days on the L1 halo (18 days on L2).
- UT containment loss: "beyond ~15 d, 70–80 %" → beyond 18–22 d, minima 68–70 %.
- Table 1: L1 halo maximum gap at 5° is 11.8 d, not 11.9 d.
- Longest gap: "8.5–16 d" → 8.5–15.7 d.
- Aperture effect: "0.3–1.6 pp" → 0.5–1.7 pp (at 5°, Tri-3+S).
- GM-UKF components: "up to 25" → up to 27.
- Prior cloud: "40 000 × 12 000 km" → 31 000 × 10 000 km (1–99 % extent).
- Search gain from 10 to 100 fields: "2–4 d" → 1.8–3.6 d.
- Velocity bound: "6–100 mm/s" → 6–101 mm/s.
- Weak direction: "7–8°" → 6.6–7.8°.
- Predictor statistics are now computed on the 394 distinct arcs. There are 525 cases because 131
  phase arcs coincide with blackout arcs.
  - UT log-R² = 0.97, 1.00 and 0.90 (was 0.98, 1.00, 0.86).
  - FTLE log-R² from −0.24 to 0.04.
- Ephemeris check: "3 of 60 outliers" → 5 of the 59 distinct arcs shift by more than 3 d.
- Tracking ANEES: both filters, not only the EKF, are slightly overconfident on the L1 halo.
- Removed, because they cannot be checked from the saved data: "converge in four iterations" and
  "247 of 250 Monte Carlo trials".
- Added: a Monte Carlo sample-size check with 2000 samples; Wilson intervals for filter success
  rates; bootstrap intervals for R²; the CRLB regularising prior.
