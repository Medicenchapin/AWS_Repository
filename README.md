# Telemarketing Assistant — Project Summary

This repository contains a prototype assistant for telemarketing teams focused on generating scripts and recommendations grounded in evidence (SHAP drivers) for prepaid customers. The project includes notebooks for EDA, feature transformation, experiments with multiple LLMs, and utilities to build prompts and call model APIs.

## Summary of Work Completed

* Translation and documentation: portions of the EDA notebook were translated, and prompts/explanations were left in English/Spanish as convenient.
* Evidence format: prompts were designed to expose per-customer SHAP-style drivers and a mirrored JSON block so the LLM generates outputs strictly based on that evidence.
* Wrappers and model experiments: calls were implemented for local models (Ollama) and provider APIs using the `together` library for TogetherAI. Cells were also prepared to compare multiple models (Hermes/Ollama, Mixtral, Gemini, Gemma).
* Credential management: the project uses `python-dotenv` and environment variables to store API keys (e.g., `TOGETHER_API_KEY` / `TOGETHERAI_API_KEY`, `OLLAMA_URL`, `OLLAMA_MODEL`).

## Repository Structure

telemarketing_assistant/

* `README.md` — this file.
* `requirements.txt` — minimal dependency list (e.g., `dotenv`, `fastapi`, `uvicorn`, `pandas`, `ollama`, `together`).
* `app/` — API/services code (e.g., `app/api/services/llm_service.py` with an example LLM HTTP call).
* `notebooks/`

  * `1_EDA.ipynb` — initial data exploration.
  * `2_FE.ipynb` — feature engineering and driver preparation.
  * `3_baseline.ipynb` — baseline notebook with prompts and examples calling Ollama and/or APIs.
  * `4_other_models.ipynb` — model comparison (Ollama/Hermes, Mixtral, Gemini, Gemma) and cells to instantiate `TogetherAIModel`.
* `scripts/`

  * `helpers.py` — utility functions to build prompts (global and per customer), SHAP driver formatting, and the features playbook.
  * `models.py` — wrappers for different LLM backends: `Model` (Ollama), `TogetherAIModel` (TogetherAI wrapper), and helper functions. Default parameters (model, temperature, max_tokens) were adjusted to obtain fuller outputs.
  * `call_togetherai.py` — (if present) example wrapper/utility for direct TogetherAI calls (see commit history if removed/restored).
* `data/` — data and feature playbooks (`data/config/feature_playbook_eng.json`, `feature_playbook_esp.json`).

## Key Notebooks and Recommended Flow

1. Open `notebooks/1_EDA.ipynb` to reproduce data exploration.
2. Run `notebooks/2_FE.ipynb` to rebuild the `drivers` column (SHAP-like list) and required artifacts.
3. Run `notebooks/3_baseline.ipynb` to see examples with Ollama and local tests.
4. Run `notebooks/4_other_models.ipynb` to compare outputs across models (Mixtral, Gemini, Gemma, and Ollama/Hermes). This notebook includes cells that instantiate `TogetherAIModel` and store responses in lists like `together_resp`, `gemini_resp`, `gemma_resp`.

Recommendation: restart the kernel before execution and run cells in order to ensure dependencies, environment variables, and objects are initialized.

## Dependencies and Installation

A `requirements.txt` with the main dependencies is provided. To create an environment and run the notebooks:

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Install the `together` library if you want the latest version (some installs are done via pip inside the notebook):

```bash
pip install together
```

3. If you will use local Ollama: follow the Ollama installation guide and ensure the service is running and the model is loaded.

## How the Models Were Tested

* `scripts/models.py` contains a `TogetherAIModel` class that uses the `together` library and expects the API key in the `TOGETHER_API_KEY` or `TOGETHERAI_API_KEY` environment variable.
* For convenience, instances were added in `notebooks/4_other_models.ipynb` for:

  * Mixtral: `mistralai/Mixtral-8x7B-Instruct-v0.1` (good quality/efficiency trade-off)
  * Gemini Pro: `google/gemini-pro` (if available on TogetherAI)
  * Gemma: `gemma/gemma-3n-e4b-instruct` (or another identifier per catalog)
  * Ollama/Hermes: local model via `ollama` (e.g., `cas/nous-hermes-2-mistral-7b-dpo`)

Each model’s responses are stored in lists (`hermes_resp`, `together_resp`, `gemini_resp`, `gemma_resp`) for comparison.
