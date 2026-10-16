.PHONY: install test lint typecheck format check generate train report serve

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src

format:
	ruff format .
	ruff check --fix .

# The same gates as CI
check: lint typecheck
	pytest --cov-fail-under=80

generate:
	fair-intervention generate --out data/raw

# Risk models + Q-learning policy into artifacts/ (needed by the API)
train:
	fair-intervention train

# All experiments -> docs/results.md and docs/img/*.png (about 3 minutes on a laptop)
report:
	fair-intervention report

serve:
	fair-intervention serve
