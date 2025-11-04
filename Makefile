.PHONY: fmt lint test

OPERON_TESTS = \
	tests/test_preprocess.py \
	tests/test_shingle.py \
	tests/test_hnsw.py \
	tests/test_sinkhorn.py \
	tests/test_graph_cluster.py \
	tests/test_cli_end2end.py

fmt:
	black operon_embed scripts $(OPERON_TESTS)
	ruff check --fix operon_embed scripts $(OPERON_TESTS)

lint:
	mypy operon_embed || true
	ruff check operon_embed scripts $(OPERON_TESTS)

default: test

test:
	pytest -q $(OPERON_TESTS)
