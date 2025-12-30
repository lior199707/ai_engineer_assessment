.PHONY: install format lint test ingest query clean

install:
	conda env update --file environment.yml --prune
	pre-commit install  # <--- Add this line here!

format:
	ruff format .

lint:
	ruff check . --fix

test:
	pytest tests/

ingest:
	python main.py ingest --data data/raw

query:
	python main.py query --q "$(Q)"

clean:
	python -c "import shutil; import pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil; import pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"