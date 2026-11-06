#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = src
PYTHON_VERSION = 0.0.1
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	pip install --target=./lib_local --no-deps -r requirements.txt


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
	PYTHONPATH=./lib_local $(PYTHON_INTERPRETER) -m src.data.dataset
	-PYTHONPATH=./lib_local $(PYTHON_INTERPRETER) -m src.creer_corpus
	PYTHONPATH=./lib_local $(PYTHON_INTERPRETER) -m src.modeling.train_bpe
	PYTHONPATH=./lib_local $(PYTHON_INTERPRETER) -m src.data.features

## Train the models (Pre-training MAE + Fine-tuning ASR (from scratch and pre-trained))
.PHONY: train
train:
	PYTHONPATH=./lib_local $(PYTHON_INTERPRETER) -m src.modeling.train
	@echo "=== Fine-tuning ASR model (From Scratch) ==="
	for seed in 42 43 44 45 46; do \
		PYTHONPATH=./lib_local $(PYTHON_INTERPRETER) -m src.modeling.train_asr --pretrained-mae-path "models/none.pt" --model-path "models/ctc_model_scratch.pt" --seed $$seed; \
	done
	@echo "=== Fine-tuning ASR model (Pre-trained) ==="
	for seed in 42 43 44 45 46; do \
		PYTHONPATH=./lib_local $(PYTHON_INTERPRETER) -m src.modeling.train_asr --pretrained-mae-path "models/mae_model.pt" --model-path "models/ctc_model_pretrained.pt" --seed $$seed; \
	done

## Test models and compute metrics (WER)
.PHONY: test
test:
	@echo "=== Evaluation: From Scratch Model ==="
	PYTHONPATH=./lib_local $(PYTHON_INTERPRETER) -m src.modeling.predict --model-path "models/ctc_model_scratch.pt"
	@echo "=== Evaluation: Pre-trained Model ==="
	PYTHONPATH=./lib_local $(PYTHON_INTERPRETER) -m src.modeling.predict --model-path "models/ctc_model_pretrained.pt"

## Run TTS Bonus model
.PHONY: bonus1
bonus1:
	@echo "=== Bonus1 (TTS) ==="
	PYTHONPATH=./lib_local $(PYTHON_INTERPRETER) -m src.modeling.train_tts
	@echo "=== Sauvegarde de l'audio  ==="
	PYTHONPATH=./lib_local python3 -m src.modeling.generate_audio

.PHONY: bonus2
bonus2:
	@echo "=== Bonus 2 (ICLR 2026 Audio-LLM) ==="
	PYTHONPATH=./lib_local $(PYTHON_INTERPRETER) -m src.modeling.train_voxtral

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
