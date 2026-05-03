# Makefile
# Maintainer: @olaflaitinen <olaf.laitinen@gmail.com>
# License: Apache-2.0
#
# Tum hedefler `uv` ve `nox` ustune bina edilmistir. Yerel makinada ve CI'de
# ayni komut calistirilir; aradaki tek fark CI'nin `--no-cache` bayragini
# eklemesidir.

SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
PYTHON ?= python3
PACKAGE := zwingen
SRC := src/$(PACKAGE)
TESTS := tests
DOCS := docs
TOOLS := tools

.PHONY: help
help:
	@awk 'BEGIN{FS=":.*?## "; printf "Hedefler:\n"} /^[a-zA-Z0-9_.-]+:.*## /{printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: sync
sync: ## uv senkronize: tum extras dahil .venv kur
	$(UV) sync --all-extras --frozen

.PHONY: lock
lock: ## uv.lock guncelle
	$(UV) lock

.PHONY: lint
lint: ## ruff check + ruff format --check
	$(UV) run ruff check $(SRC) $(TESTS) $(TOOLS)
	$(UV) run ruff format --check $(SRC) $(TESTS) $(TOOLS)

.PHONY: format
format: ## ruff format ve ruff check --fix
	$(UV) run ruff format $(SRC) $(TESTS) $(TOOLS)
	$(UV) run ruff check --fix $(SRC) $(TESTS) $(TOOLS)

.PHONY: typecheck
typecheck: ## mypy strict + pyright strict
	$(UV) run mypy $(SRC) $(TOOLS)
	$(UV) run pyright

.PHONY: style-guard
style-guard: ## emoji ve em/en-dash yasagini denetle (README, docs, public surface)
	$(UV) run python $(TOOLS)/lint_no_emoji.py README.md $(DOCS)
	$(UV) run python $(TOOLS)/check_dashes.py README.md $(DOCS)

.PHONY: test
test: ## hizli pytest (slow ve gpu hariclendi)
	$(UV) run pytest -m "not slow and not gpu" -n auto

.PHONY: test-all
test-all: ## tum pytest, kapsam kontrolu dahil
	$(UV) run pytest

.PHONY: reproducibility
reproducibility: ## reproducibility receipt karsilastirma
	$(UV) run pytest tests/reproducibility -m reproducibility

.PHONY: hypothesis
hypothesis: ## hypothesis senaryo profili
	$(UV) run pytest -m hypothesis --hypothesis-show-statistics

.PHONY: bench
bench: ## asv ile mikrobenchmark
	$(UV) run asv run --quick HEAD^!

.PHONY: docs
docs: ## yerel mkdocs serve
	$(UV) run mkdocs serve --strict --watch $(SRC)

.PHONY: docs-build
docs-build: ## strict mkdocs build (ReadTheDocs ile ayni)
	$(UV) run mkdocs build --strict

.PHONY: build
build: ## sdist + wheel (yerel hatchling)
	$(UV) run python -m build --sdist --wheel

.PHONY: wheels
wheels: ## cibuildwheel ile platform tekerlekleri
	$(UV) run python -m cibuildwheel --output-dir dist

.PHONY: receipts
receipts: ## ornek reproducibility receipt karsilastir
	$(UV) run python $(TOOLS)/compare_receipts.py \
		tests/reproducibility/fixtures/receipt_baseline.json \
		tests/reproducibility/fixtures/receipt_candidate.json \
		--regime tolerant

.PHONY: changelog
changelog: ## CHANGELOG.md uret
	$(UV) run python $(TOOLS)/generate_changelog.py --in-place

.PHONY: precommit
precommit: ## pre-commit tum dosyalar
	$(UV) run pre-commit run --all-files --show-diff-on-failure

.PHONY: nox
nox: ## kanonik nox oturumu
	$(UV) run nox -s tests typecheck lint docs

.PHONY: ci
ci: lint typecheck style-guard test reproducibility docs-build ## CI ile ayni hedef zinciri

.PHONY: clean
clean: ## build ciktisi, cache, ve kapsama dosyalarini sil
	rm -rf build/ dist/ wheelhouse/ .pytest_cache/ .mypy_cache/ .ruff_cache/ \
	       .nox/ .coverage coverage.xml htmlcov/ site/ \
	       $(SRC).egg-info/ src/*.egg-info/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
