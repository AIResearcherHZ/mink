.PHONY: sync
sync:
	uv sync --all-extras --all-packages --group dev

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: type
type:
	uv run ty check
	uv run pyright

.PHONY: check
check: format type

.PHONY: test
test:
	uv run pytest

.PHONY: test-all
test-all: check test

.PHONY: coverage
coverage:
	uv run coverage run --source=src -m pytest .
	uv run coverage report

.PHONY: doc
doc:
	uv run --group doc sphinx-build docs _build -W

.PHONY: build
build:
	rm -rf dist/
	uv build
	uv run --no-cache --isolated --no-project --with pytest --with robot_descriptions --with dist/*.whl pytest tests/
	uv run --no-cache --isolated --no-project --with pytest --with robot_descriptions --with dist/*.tar.gz pytest tests/
	@ls -lh dist/*.whl | awk '{print "Wheel size: " $$5}'
	@echo "Build and test successful"
