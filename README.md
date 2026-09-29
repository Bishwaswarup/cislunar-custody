# cislunar-custody

**Custody horizons for ground-based, angles-only tracking of cislunar periodic orbits.**

Suppose Earth-based optical telescopes track an object on a halo, NRHO, DRO or Lyapunov
orbit, and then lose it. How long can the object go unobserved before it falls outside the
telescope search pattern for good? And can that *custody horizon* be predicted without
Monte Carlo simulation?

Author: Bishwaswarup Nayak (Indian Institute of Science). Code: MIT License. Manuscript draft: `paper/`.

## Headline results

Reference setup: 1 m telescopes at Hanle, La Palma, Haleakala and Siding Spring; 5° Moon
exclusion; 1″ angle noise; 2 m target (albedo 0.2); search patterns of 1, 10 or 100
one-degree fields.

| Result | Numbers | Figure |
| --- | --- | --- |
| Cislunar targets sit close to the Moon as seen from Earth | L1 halo 3.9–6.1°, L2 Lyapunov 0–7°, 9:2 NRHO 0.5–10°, DRO 0–14° | fig05 |
| The Moon-exclusion angle matters more than aperture | 10° exclusion: L1/L2 targets never visible; 5°: 21–33 % of the year | fig03 |
| A lunar-phase blackout is unavoidable | longest gap 8.5–16 d for every network and aperture | fig04 |
| Tracking accuracy after 7-day arcs (CRLB, reached by the filters) | NRHO ≈ 1 km, L1 halo ≈ 2 km, DRO ≈ 7 km, L2 Lyapunov ≈ 17 km | fig06, fig08 |
| Reacquisition after a 3.2 d gap (L1 halo): share of consistent runs | EKF 10 %, UKF 70 %, PF→UKF 70 %, GM-UKF 100 % | fig10, fig11 |
| Custody horizon, 10 fields, after 1 / 3 / 7-day arcs | NRHO and DRO > 30 d; L1 halo 11 / 13 / 17 d; L2 Lyapunov 12 / 15 / 17 d | fig12 |
| Blackouts survived after 1 / 3 / 7-day arcs (10 fields) | 74–89 % / 96–100 % / 100 % | fig15 |
| Predicting T_c without Monte Carlo (log R²) | unscented transform 0.86–1.00, linear 0.90–0.98, orbit-only FTLE ≤ 0.03 | fig13 |
| Linear (EKF-style) prediction gives false custody | L1 halo after 14 d: the linear 99 % ellipse holds 7.6 % of the true mass, the UT ellipse 98 % | fig14 |
| The results hold in DE440 + SRP dynamics | T_c ratio ephemeris/CR3BP 0.93–1.01; growth curves overlap | fig16, fig17 |

**Operational rules.** Track for at least 3 days (preferably 7) before a predicted
blackout. Use unscented or Gaussian-mixture prediction, not linear prediction, for gaps
longer than about a week on unstable orbits.

## Status

| Milestone | What | Status |
| --- | --- | --- |
| 1 | CR3BP dynamics, STM, periodic-orbit catalogue | done |
| 2 | Ground sensor model and one-year visibility | done |
| 3 | Angles-only observability (Fisher information, CRLB) | done |
| 4 | EKF and UKF: Monte Carlo consistency, track vs reacquire | done |
| 5 | GM-UKF and PF→UKF for post-gap reacquisition | done |
| 6 | Custody horizons (linear / UT / Monte Carlo), predictors, blackouts | done |
| 7 | DE440 ephemeris + SRP cross-check with multiple-shooting counterparts | done |
| 8 | Manuscript draft (`paper/`) | draft |
| — | Real-data validation with Indian observatories (IIA, ARIES, GROWTH-India) | phase 2 |

## Install and test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ephem]"
curl -o data/de440s.bsp https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp
pytest                                    # 47 tests, ~10 s (ephemeris tests skip without the kernel)
```

## Reproduce every result (in order)

| Step | Command | Time | Outputs |
| --- | --- | --- | --- |
| 1 | `python scripts/build_catalogue.py` | ~15 s | `data/catalogue.npz`, `data/named_orbits.json` |
| 1 | `python scripts/plot_catalogue.py` | ~5 s | fig01, fig02 |
| 2 | `python scripts/visibility_study.py` | ~10 s | `data/visibility_summary.csv`, fig03–fig05 |
| 3 | `python scripts/observability_study.py` | ~10 s | `data/observability_arcs.csv`, fig06, fig07 |
| 4 | `python scripts/filter_study.py` | ~1 min | `data/filter_summary.csv`, fig08, fig09 |
| 5 | `python scripts/nonlinear_filter_study.py` | ~1 min | `data/nonlinear_filter_summary.csv`, fig10, fig11 |
| 6 | `python scripts/custody_study.py` | ~12 min | `data/custody_cases.csv`, fig12–fig15 |
| 7 | `python scripts/ephemeris_check.py` | ~6 min | `data/ephemeris_check.csv`, fig16, fig17 |

Each script prints its summary tables. Steps 3–7 depend on the catalogue, and steps 4–7
also read the CSVs from earlier steps.

## Layout

```
src/cislunar_custody/
  constants.py, timeutil.py      DE440 GM values, LU/TU, radii; Julian dates
  dynamics/   cr3bp.py           EOM, STM, Jacobi constant, Lagrange points, vectorised propagation
              periodic.py        differential correction, pseudo-arclength continuation, PeriodicOrbit
              seeds.py           Richardson 3rd-order halo, linear Lyapunov, DRO initial guesses
  catalogue.py                   build / save / load the orbit families (L1/L2 halo, NRHO, Lyapunov, DRO)
  frames/                        low-precision Sun/Moon, GMST, sites, CR3BP -> inertial mapping
  sensors/                       sites, telescopes, photometry, Krisciunas-Schaefer sky, visibility, RA/Dec
  observability/fisher.py        Fisher information and CRLB for real measurement schedules
  filters/                       EKF, UKF, GM-UKF (entropy-triggered splitting), PF with UKF handover
  scenario.py                    measurement generation and a filter runner shared by all filters
  custody/horizon.py             gap propagation (linear / UT / MC), sky metrics, custody horizons
  ephem/                         DE440 (jplephem), Earth-centred ephemeris + SRP dynamics, exact STM,
                                 CR3BP <-> inertial transform, multiple shooting (LM + Newton)
scripts/                         one script per milestone (see table above)
tests/                           47 pytest tests
data/, figures/                  generated outputs
paper/                           LaTeX manuscript (main.tex, references.bib)
```

## Modelling assumptions (stated in the paper)

- Truth and filters share the dynamics. The v1 sky geometry places the CR3BP orbit in the
  real Earth–Moon direction (analytic Sun and Moon; the DE440 check confirms the results).
- Telescope limiting magnitudes (19.5 / 21.5 / 23.0 for 0.36 / 1 / 2 m), extinction and
  dark-sky brightness are nominal assumptions. The first is anchored to LANL's 36 cm
  detection of CAPSTONE.
- The post-fit uncertainty at the start of a gap is the CRLB of the preceding arc.
  Milestone 4 shows that the filters reach it.
- The NRHO ephemeris counterpart keeps a 50–124 km continuity residual, because
  perilune conditioning stalls the corrector.

## Citation

Manuscript in preparation (see `paper/`). Please contact the author before reusing the results.
