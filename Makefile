# Rowan — single entry points for the hub, the room client, tests and migrations
# (ТЗ phase 0). On Windows run these from Git Bash / MSYS make, or use the
# underlying `python -m ...` commands directly (see docs/ARCHITECTURE.md).

PY ?= python
TESTS ?= tests
CONFIG ?= config.yaml

.PHONY: hub client test test-regress migrate skill help

help:
	@echo "make hub           - run the hub (brain) server"
	@echo "make client        - run the room client"
	@echo "make test          - run the full pytest suite"
	@echo "make test-regress  - run only the utterance regression harness"
	@echo "make migrate       - apply pending SQLite migrations to data/hub.db"
	@echo "make skill name=X  - scaffold a new skill in skills/X"

hub:
	$(PY) -m hub.main --config $(CONFIG)

client:
	$(PY) -m client.main --config $(CONFIG)

test:
	$(PY) -m pytest $(TESTS) -q

test-regress:
	$(PY) -m pytest tests/regress -q

migrate:
	$(PY) -m hub.migrations_runner --db data/hub.db

skill:
	@test -n "$(name)" || (echo "usage: make skill name=<name>" && exit 2)
	$(PY) -m hub.skill_scaffold --name $(name)
