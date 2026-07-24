PYTHON ?= python
CORE_PYTHON := .venv/bin/python
GEDI_PYTHON := .venv-gedi/bin/python
MANIFEST_BUILD_ROOT ?= build/manifests/v6

.PHONY: env verify-inputs data-smoke data-all manifests manifests-survey manifests-hpo manifests-all check submission-archive

env:
	$(PYTHON) -m venv .venv
	$(CORE_PYTHON) -m pip install -r requirements.txt
	$(CORE_PYTHON) -m pip install --no-deps -e .
	$(PYTHON) -m venv .venv-gedi
	$(GEDI_PYTHON) -m pip install -r environments/gedi/requirements.txt

verify-inputs:
	$(CORE_PYTHON) scripts/verify_inputs.py --all

data-smoke:
	$(CORE_PYTHON) -m pytest -q tests/test_data_pipeline_integration.py

data-all: verify-inputs
	$(CORE_PYTHON) scripts/augment_logs.py --all
	$(CORE_PYTHON) scripts/generate_feature_space_logs.py --mode design --n-targets 200 --compute-anchor
	$(CORE_PYTHON) scripts/generate_feature_space_logs.py --mode main --n-targets 200 --gedi-python $(GEDI_PYTHON)
	$(CORE_PYTHON) scripts/verify_generated_data.py

manifests:
	$(CORE_PYTHON) scripts/generate_v6_manifests.py --primary --output-root $(MANIFEST_BUILD_ROOT)
	$(CORE_PYTHON) scripts/manifest_receipts.py --check --scope primary --manifest-root $(MANIFEST_BUILD_ROOT)

manifests-survey:
	$(CORE_PYTHON) scripts/generate_v6_manifests.py --default-run-survey --output-root $(MANIFEST_BUILD_ROOT)
	$(CORE_PYTHON) scripts/manifest_receipts.py --check --scope survey --manifest-root $(MANIFEST_BUILD_ROOT)

manifests-hpo:
	@echo "WARNING: HPO is deprecated legacy tooling and is not required for submission."
	set -e; for config in configs/experiments/v6/hpo/*/*.yaml; do \
		$(CORE_PYTHON) scripts/generate_hpo_studies.py --config "$$config" --output-root $(MANIFEST_BUILD_ROOT); \
	done
	$(CORE_PYTHON) scripts/manifest_receipts.py --check --scope hpo --manifest-root $(MANIFEST_BUILD_ROOT)

manifests-all:
	$(CORE_PYTHON) scripts/generate_v6_manifests.py --all --output-root $(MANIFEST_BUILD_ROOT)
	$(CORE_PYTHON) scripts/manifest_receipts.py --check --scope primary --manifest-root $(MANIFEST_BUILD_ROOT)
	$(CORE_PYTHON) scripts/manifest_receipts.py --check --scope survey --manifest-root $(MANIFEST_BUILD_ROOT)

check:
	$(CORE_PYTHON) -m ruff check .
	$(CORE_PYTHON) -m pytest -m "not external and not legacy_hpo"
	$(CORE_PYTHON) scripts/manifest_receipts.py --check --scope primary
	$(CORE_PYTHON) scripts/audit_submission.py

submission-archive:
	$(CORE_PYTHON) scripts/build_submission_archive.py
