#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = src
PYTHON_VERSION = 0.0.1
PYTHON_INTERPRETER ?= python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 src

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset and extract features
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m src.data.dataset
	$(PYTHON_INTERPRETER) -m src.creer_corpus
	$(PYTHON_INTERPRETER) -m src.modeling.train_bpe
	$(PYTHON_INTERPRETER) -m src.data.features

## Train the models (Pre-training MAE + Fine-tuning ASR (from scratch and pre-trained))
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m src.modeling.train
	@echo "=== Fine-tuning ASR model (From Scratch) ==="
	for seed in 42 43 44 45 46; do \
		$(PYTHON_INTERPRETER) -m src.modeling.train_asr --pretrained-mae-path "models/none.pt" --model-path "models/ctc_model_scratch.pt" --seed $$seed; \
	done
	@echo "=== Fine-tuning ASR model (Pre-trained) ==="
	for seed in 42 43 44 45 46; do \
		$(PYTHON_INTERPRETER) -m src.modeling.train_asr --pretrained-mae-path "models/mae_model.pt" --model-path "models/ctc_model_pretrained.pt" --seed $$seed; \
	done

## Test models and compute metrics (WER)
.PHONY: test
test:
	@echo "=== Evaluation: From Scratch Model ==="
	$(PYTHON_INTERPRETER) -m src.modeling.predict --model-path "models/ctc_model_scratch.pt"
	@echo "=== Evaluation: Pre-trained Model ==="
	$(PYTHON_INTERPRETER) -m src.modeling.predict --model-path "models/ctc_model_pretrained.pt"

## Run TTS Bonus model
.PHONY: bonus1
bonus1:
	@echo "=== Bonus1 (TTS) ==="
	$(PYTHON_INTERPRETER) -m src.modeling.train_tts
	@echo "=== Sauvegarde de l'audio  ==="
	$(PYTHON_INTERPRETER) -m src.modeling.generate_audio

.PHONY: bonus2
bonus2:
	@echo "=== Bonus 2 (ICLR 2026 Audio-LLM) ==="
	$(PYTHON_INTERPRETER) -m src.modeling.train_voxtral

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
