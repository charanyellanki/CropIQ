.PHONY: install data train serve dashboard test gate clean all

PYTHON ?= python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.data.fetch_nass
	$(PYTHON) -m src.data.fetch_noaa
	$(PYTHON) -m src.data.fetch_ssurgo
	$(PYTHON) -m src.data.build_dataset

train:
	$(PYTHON) -m src.train
	$(PYTHON) scripts/build_evaluation_artifacts.py

serve:
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(PYTHON) -m streamlit run app.py

test:
	$(PYTHON) -m pytest -v --cov=src --cov-report=term-missing tests/

gate:
	$(PYTHON) scripts/run_gate.py $(PHASE)

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: install data train test
	$(PYTHON) scripts/run_gate.py 0
	$(PYTHON) scripts/run_gate.py 2
	$(PYTHON) scripts/run_gate.py 4
	$(PYTHON) scripts/run_gate.py 5
	$(PYTHON) scripts/run_gate.py 6
	$(PYTHON) scripts/run_gate.py 7
	$(PYTHON) scripts/run_gate.py 9
