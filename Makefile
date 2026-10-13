.PHONY: install test lint synth run api dash docker-build clean

PY ?= python

install:
	$(PY) -m pip install -e ".[dashboard,dev]"

test:
	$(PY) -m pytest -q

lint:
	ruff check .

# simulated M5-shaped data for development and CI (item ids contain SIM, everything is labelled)
synth:
	$(PY) scripts/make_synthetic.py --out data/synthetic

# real data: pass the folder or the m5_foods.zip produced by scripts/prepare_m5_upload.py
DATA ?= data/m5_foods.zip
run:
	$(PY) scripts/run_pipeline.py --data $(DATA) --results-dir results

api:
	uvicorn stockwise.api:app --host 0.0.0.0 --port 8000

dash:
	streamlit run app/dashboard.py

docker-build:
	docker build -t stockwise .

clean:
	rm -rf .pytest_cache .ruff_cache
