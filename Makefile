.PHONY: install format lint test ingest query clean

install:
	conda env update --file environment.yml --prune
	pre-commit install

format:
	ruff format .

lint:
	ruff check . --fix

test:
	python -m pytest tests/

# UPDATED: Runs as a module (src.main)
# Usage: make ingest data="path/to/docs"
ingest:
	python -m src.main ingest --data $(or $(data),data/raw)

# UPDATED: Runs as a module (src.main)
# Usage: make query Q="Your question here"
query:
	python -m src.main query --q "$(Q)"

clean:
	python -c "import shutil; import pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil; import pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"

run:
	python -m uvicorn src.api.main:app --reload --port 8000