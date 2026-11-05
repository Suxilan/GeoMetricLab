PYTHON ?= python
CONDA ?= conda
ENV_NAME ?= GeoMetricLab
EXTRAS ?= runtime,models,peft,artifacts,swanlab
PIP_FLAGS ?=

.PHONY: help install develop conda-env repair-env check-env

help:
	@echo "make conda-env  Create the Linux/NVIDIA Conda environment and configure its runtime"
	@echo "make install    Install the library and selected extras into the current Python"
	@echo "make develop    Install editable sources and selected extras"
	@echo "make repair-env Configure C++ runtime loading in the current Conda environment"
	@echo "make check-env  Check imports, dependencies and available FAISS backends"

install:
	$(PYTHON) -m pip install $(PIP_FLAGS) ".[${EXTRAS}]"
	$(PYTHON) tools/environment/configure_conda.py

develop:
	$(PYTHON) -m pip install $(PIP_FLAGS) -e ".[${EXTRAS}]"
	$(PYTHON) tools/environment/configure_conda.py

conda-env:
	$(CONDA) env create --name "$(ENV_NAME)" --file environment.yml
	$(CONDA) run --name "$(ENV_NAME)" python tools/environment/configure_conda.py
	@echo "Run: conda activate $(ENV_NAME)"

repair-env:
	$(PYTHON) tools/environment/configure_conda.py

check-env:
	$(PYTHON) -I -m pip check
	$(PYTHON) -I tools/environment/check.py
