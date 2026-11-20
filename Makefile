install:
	pip install -r requirements.txt && pip install -e . --no-deps

lint:
	ruff check .

test:
	pytest -q

run:
	PYTHONPATH=src python -m frankfurt_rent.pipeline

clean:
	rm -rf data reports .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: install lint test run clean
