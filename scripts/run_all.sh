#!/usr/bin/env bash
# Re-run the whole study, regenerate every figure and table, and check the paper's numbers.
#   bash scripts/run_all.sh            # full run (~100 min on a laptop)
#   bash scripts/run_all.sh --check    # only re-check the paper against the existing data/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

if [[ "${1:-}" != "--check" ]]; then
  python -m pytest -q                                 2>&1 | tee logs/00_pytest.log
  python scripts/build_catalogue.py                   2>&1 | tee logs/01_catalogue.log
  python scripts/plot_catalogue.py                    2>&1 | tee logs/01b_plot_catalogue.log
  python scripts/visibility_study.py                  2>&1 | tee logs/02_visibility.log
  python scripts/observability_study.py               2>&1 | tee logs/03_observability.log
  python scripts/filter_study.py                      2>&1 | tee logs/04_filters.log
  python scripts/nonlinear_filter_study.py            2>&1 | tee logs/05_nonlinear.log
  python scripts/custody_study.py                     2>&1 | tee logs/06_custody.log
  python scripts/ephemeris_check.py                   2>&1 | tee logs/07_ephemeris.log
  python scripts/sensitivity_check.py                 2>&1 | tee logs/08_sensitivity.log
  python scripts/family_sweep.py                      2>&1 | tee logs/10_family_sweep.log
  python scripts/consider_study.py                    2>&1 | tee logs/11_consider.log
fi
python scripts/verify_claims.py                       2>&1 | tee logs/09_verify_claims.log
