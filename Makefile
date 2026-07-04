SHELL := /bin/bash
.DELETE_ON_ERROR:
.DEFAULT_GOAL := help

PY := PYTHONPATH=src python3

# --- gathering knobs (override on the command line) -------------------------
PROVIDER ?= github
TARGET   ?= 500
SINCE    ?= 2019-01-01
QUERY    ?= stars:>100 pushed:>2024-01-01
LIST     ?=
CAP_GB   ?= 3
CONC     ?=

# list provider needs --list; others ignore it
LIST_ARG := $(if $(LIST),--list $(LIST),)
CONC_ARG := $(if $(CONC),--max-concurrency $(CONC),)

# --- cache stamps: presence means "already done", so nothing re-runs until
#     `make data-clear` (records) or `make results-clear` (derived) removes them
RECORDS_STAMP := data/records/.gathered
PANEL_STAMP   := data/panels/.built
RESULTS_STAMP := data/results/.analysed

.PHONY: help all get-data analyse analyze data-clear results-clear \
        gather synth panel stats plots manifest test typecheck clean distclean

help: ## show available targets
	@echo "High level:"
	@grep -hE '^[a-zA-Z_-]+:.*?##@ ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?##@ "}{printf "  %-14s %s\n", $$1, $$2}'
	@echo "Low level:"
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

# ============================================================================
# High-level commands
# ============================================================================

all: get-data analyse ##@ gather (if needed) then run the full analysis

get-data: $(RECORDS_STAMP) ##@ collect git history (skips if data already present)

analyse: $(RESULTS_STAMP) plots manifest ##@ build panels, run stats, render graphs
analyze: analyse  ## alias for analyse

data-clear: ##@ delete ALL collected + derived data (forces a fresh fetch next time)
	rm -rf data/records/records-*.jsonl data/records/failures-*.jsonl $(RECORDS_STAMP)
	rm -rf data/panels/* data/results/* data/raw-cache/*

results-clear: ##@ delete only derived panels/results (keeps raw records)
	rm -rf data/panels/* data/results/* $(PANEL_STAMP) $(RESULTS_STAMP)

# ============================================================================
# Stamped pipeline stages (cached; edit deps -> only affected stages re-run)
# ============================================================================

$(RECORDS_STAMP):
	$(PY) -m aisloc.gather --provider $(PROVIDER) $(LIST_ARG) $(CONC_ARG) \
		--target $(TARGET) --since $(SINCE) --per-repo-cap-gb $(CAP_GB) \
		--query '$(QUERY)'
	@touch $@

$(PANEL_STAMP): $(RECORDS_STAMP)
	$(PY) -m aisloc.analysis.panel
	@touch $@

$(RESULTS_STAMP): $(PANEL_STAMP)
	$(PY) -m aisloc.analysis.stats
	@touch $@

# ============================================================================
# Low-level commands (always run; ignore the cache)
# ============================================================================

gather: ## force a gather run now (bypasses the cache stamp)
	$(PY) -m aisloc.gather --provider $(PROVIDER) $(LIST_ARG) $(CONC_ARG) \
		--target $(TARGET) --since $(SINCE) --per-repo-cap-gb $(CAP_GB) --query '$(QUERY)'
	@touch $(RECORDS_STAMP)

synth: ## generate a synthetic panel with a planted AI effect (for validation)
	$(PY) -m aisloc.analysis.synth
	@touch $(RECORDS_STAMP)

panel: ## consolidate records into tidy CSV/Parquet panels
	$(PY) -m aisloc.analysis.panel

stats: ## run the statistical stack (excess churn, dose-response, PU propensity)
	$(PY) -m aisloc.analysis.stats

plots: ## render matplotlib figures into data/results/
	$(PY) -m aisloc.analysis.plots

manifest: ## write per-repo audit manifest (uses model p_ai if analysis ran)
	$(PY) -m aisloc.analysis.manifest

test: ## run unit tests
	$(PY) -m pytest -q

typecheck: ## strict mypy over the package
	$(PY) -m mypy --strict src/aisloc

clean: ## remove clone scratch and derived results (keeps records)
	rm -rf data/raw-cache/* data/panels/* data/results/* $(PANEL_STAMP) $(RESULTS_STAMP)

distclean: data-clear ## full reset to a fresh checkout state
	rm -rf data/**/__pycache__ src/**/__pycache__
