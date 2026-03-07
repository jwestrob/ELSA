.PHONY: fmt lint test smoke

fmt:
	ruff check --fix elsa/ genome_browser/ benchmarks/scripts/
	ruff format elsa/ genome_browser/ benchmarks/scripts/

lint:
	ruff check elsa/ genome_browser/ benchmarks/scripts/
	python -m compileall elsa genome_browser benchmarks/scripts -q

test:
	pytest -q tests/

smoke:
	python -m compileall elsa -q
	elsa --help > /dev/null
	elsa analyze --help > /dev/null
	elsa search --help > /dev/null
