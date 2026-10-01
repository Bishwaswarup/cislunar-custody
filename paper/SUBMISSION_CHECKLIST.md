# JAS submission checklist

Target journal: *The Journal of the Astronautical Sciences* (Springer). Build the PDF with
`pdflatex main && bibtex main && pdflatex main && pdflatex main` in `paper/`. The build needs
`sn-jnl.cls` and `sn-mathphys-num.bst`, which are in this folder, and `../figures/jas/`.

## Done in the manuscript

- [x] Springer Nature template (`sn-jnl`), numbered references in square brackets (`sn-mathphys-num`).
- [x] Title page: author, department, institute, city, postcode, country, e-mail, ORCID.
- [x] Abstract of about 248 words (limit 150–250; recount after any edit). Abbreviations are spelled out.
- [x] Six keywords.
- [x] At most three heading levels, all numbered.
- [x] Declarations: funding, competing interests, data and code availability, author
      contributions, ethics.
- [x] Related work expanded (1 Oct): 35 references; the 10 new ones (cislunar SDA surveys, space- and ground-based observation geometry, uncertainty propagation and realism) were checked against Crossref, DTIC and AFRL.
- [x] References have DOIs where one exists. They were checked against Crossref and publisher
      pages. Jo et al. is updated to the 2026 *Astrodynamics* paper, and Iannamorelli & LeGrand to
      JAS 72(1) 2025.
- [x] Figures are `figures/jas/Fig1…Fig16` (Fig14–15 from `family_sweep.py`, Fig16 from `consider_study.py`), in vector PDF and 600 dpi TIFF:
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
      `sn-mathphys-num.bst`, `Fig1–Fig16` (PDF or TIFF) and the compiled `main.pdf`.

## Evidence that is right but statistically thin (`verify_claims.py` shows WEAK)

| Claim | Why it is thin | Where it is disclosed |
| :-- | :-- | :-- |
| UT and linear predictor log-R² = 0.96 at N = 1 | bootstrap 95 % CI 0.87–1.00 | Sec. 8 gives the interval |
| UT predictor log-R² = 0.93 at N = 100 | bootstrap 95 % CI 0.85–0.98 | Sec. 8 gives the interval |
| Ephemeris UT log-R² = 0.90 at N = 1 | 42 finite cases, CI 0.59–1.00 | Sec. 9 gives n and the interval |
| Ephemeris UT log-R² = 0.80 at N = 100 | 30 finite cases, CI 0.55–0.99 | Sec. 9 gives n and the interval |
| Ephemeris, L1 halo, 1-day arcs | only 4 cases | Table 5 column n |
| L2 halo family: ρ = −0.61, p = 0.14 | 7 members, 6 of them beyond 30 d | Sec. 10.1 states it openly |
| Strip gains (cross-track 0.5″ vs 1.4″) | rest on the CRLB covariance without biases | Sec. 10.2 and Limitations; milestone 9 re-tests |

## Milestone 9 and the noise-model fix (this version)

- Noise model: 1″ isotropic on the sky everywhere. The simulator already did this; the CRLB and the
  filters used 1″ in right ascension itself. All CRLB-based numbers moved by about 1–3 % (Tables 2, 3, 5,
  Sec. 6–10). The L1 tracking overconfidence (ANEES 9.1 / 7.7) disappeared; only the EKF on L2 is outside
  the band now (9.25).
- Reacquisition arcs are now chosen explicitly (start at the observation gap closest to 3 d); before, the
  3.2-d gap came from whichever arc had the median CRLB. L1: EKF 37 %, UKF 77 %, PF→UKF 88 %, GM-UKF 100 %.
- New Sec. 11 "Robustness to Unmodelled Errors" (consider covariance, Fig. 16, verifier claims C1–C11),
  Discussion rule 6, contribution 6, abstract and conclusions sentences.
- WEAK: L2 halo family ρ (FS4b), strip cross-track width (FS6g), predictor and ephemeris R² with wide CIs.

## Milestone 8 in the paper: operator horizon as the headline

- Sec. 4 defines the **operator horizon** (circle centred on the UT prediction). Tables 3 and 4 now
  report it; the ideal horizon agrees to within 0.4 d (N ≤ 10) and 0.9 d (N = 100), median ratio 1.00.
- Table 3 cells that moved (ideal → operator): L1 1-d N=10 11.0 → 10.9, N=100 12.8 → 12.7; L2 1-d
  N=100 16.2 → 15.8; L2 3-d N=100 18.3 → 18.0; L2 7-d N=100 19.2 → 19.1; some percentiles by 0.1–0.3 d.
- Table 4: the separate "UT-centred" column is gone; operator and ideal survival shares are identical.
- Sec. 7 text: 13.9–19.2 → 13.9–19.1 d; gain 10 → 100 fields 1.7–3.6 → 1.7–3.3 d; 11.0–17.4 → 10.9–17.4 d.
- New Sec. 10 "Horizon Across Orbit Families" with Fig. 14 (T_c vs ν) and Fig. 15 (operator vs ideal,
  strip). New verifier claims FS1–FS6 (named FS, because F1/F2 are already the filter claims) and K14.
- Sec. 8: "a linear predictor ... would point the telescopes at the wrong part of the sky" was too
  strong (a circle centred on the linear prediction loses custody at nearly the same time as the
  operator circle). Now: the linear failure lies in the predicted spread, not the predicted position.
- Abstract, contributions, Discussion and Conclusions lead with the operator horizon and the family result.

## Numbers that changed with the 121-point gap grid (m8-fixes)

The gap grid went from 37 to 121 points (0 plus 120 log-spaced values in 0.1–30 d). The 37-point
grid missed the NRHO perilune spikes in the sky radius. A uniform 0.02-d grid changes T_c by at most
0.15 d (N = 1, 10) and 0.6 d (N = 100), so the 121-point grid is converged.

- Table 3: most cells move by 0.1–0.7 d. NRHO 1-d N = 1 lower percentile 17.2 → 5.3 d (perilune).
- Table 4: NRHO 1-d N = 1 survival 78 → 67 %; DRO 1-d N = 1 54 → 46 %.
- Table 5: NRHO 1-d ephemeris median >30 → 28.3 d; small shifts elsewhere.
- UT false custody: "never" → 2 of 254 distinct arcs.
- Linear containment at 14 d (L1): 7.6 → 5.4 %. Linear < 95 %: 9.6 → 9.5 d (L1), 18.4 → 18.6 d (L2).
- UT containment minima 68–70 → 66–72 %, first below 95 % at 18.6–19.5 d.
- Predictor log-R²: UT 0.96 / 1.00 / 0.93, linear 0.96 / 0.99 / 0.97; FTLE −0.23 to 0.04.
- Ephemeris: 6 (was 5) of 59 arcs shift > 3 d; UT log-R² 0.90 / 1.00 / 0.80; linear containment
  "worse in the ephemeris model" now holds only for L2 (84.8 % vs 98.6 %); L1 is 12.6 % vs 10.3 %.
- Sample size: N = 10, 100 at most 0.9 d; the N = 1 worst case is an NRHO arc at perilune (5.3 → 17.6 d).
- New rule 5 in the Discussion: time narrow NRHO searches away from perilune.

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
