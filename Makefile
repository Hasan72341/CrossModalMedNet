PYTHON ?= python3
PROJECT_DIR := project-group-5
MANIFEST_PATH ?= SynthRAD2023_Dataset/manifest.csv
TEST_LIST ?= $(PROJECT_DIR)/scripts/test_list.txt

.PHONY: install install-dev syntax check-paths verify-checkpoints smoke-pix2pix-brain clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(PROJECT_DIR)/requirements.txt
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,metrics]"

syntax:
	$(PYTHON) -m py_compile \
		$(PROJECT_DIR)/scripts/evaluate_all_models.py \
		$(PROJECT_DIR)/scripts/smoke_eval.py \
		$(PROJECT_DIR)/scripts/test_dataloaders.py \
		$(PROJECT_DIR)/scripts/verify_checkpoints.py \
		$(PROJECT_DIR)/scripts/generate_servers.py \
		$(PROJECT_DIR)/generate_specific_visuals.py \
		$(PROJECT_DIR)/test_dataset.py \
		$(PROJECT_DIR)/inspect_data.py \
		$(PROJECT_DIR)/start_all.py \
		$(PROJECT_DIR)/test_crash.py

check-paths:
	! rg -n "$$(printf '/%s|/%s' usershome home)" $(PROJECT_DIR)/scripts $(PROJECT_DIR)/*.py README.md REPRODUCIBILITY.md CONTRIBUTING.md

verify-checkpoints:
	cd $(PROJECT_DIR) && $(PYTHON) scripts/verify_checkpoints.py

smoke-pix2pix-brain:
	cd $(PROJECT_DIR) && $(PYTHON) scripts/smoke_eval.py \
		--model_name pix2pix_brain \
		--model_type pix2pix \
		--region brain \
		--ckpt_dir models/pix2pix/brain/checkpoints \
		--manifest_path ../$(MANIFEST_PATH) \
		--test_list scripts/test_list.txt

clean:
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
