#!/bin/sh
set -e
if [ ! -f "$RESULTS_DIR/meta.json" ]; then
  echo "No results in $RESULTS_DIR: generating a SIMULATED demo (not real data)."
  python scripts/make_synthetic.py --out /tmp/sim --items-per-dept 10 --days 900 >/dev/null
  python scripts/run_pipeline.py --data /tmp/sim --results-dir "$RESULTS_DIR" \
    --val-days 30 --test-days 60 --train-window 300 --n-estimators 30 --no-figures
fi
case "$1" in
  api) exec uvicorn stockwise.api:app --host 0.0.0.0 --port 8000 ;;
  dashboard) exec streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0 ;;
  *) exec "$@" ;;
esac
