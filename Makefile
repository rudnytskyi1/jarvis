# Rowan — single entry points for the hub, the room client, tests and migrations
# (ТЗ phase 0). On Windows run these from Git Bash / MSYS make, or use the
# underlying `python -m ...` commands directly (see docs/ARCHITECTURE.md).

PY ?= python
TESTS ?= tests
CONFIG ?= config.yaml

.PHONY: hub client test test-regress test-identity measure-scene measure-presence migrate skill help

help:
	@echo "make hub           - run the hub (brain) server"
	@echo "make client        - run the room client"
	@echo "make test          - run the full pytest suite"
	@echo "make test-regress  - run only the utterance regression harness"
	@echo "make test-identity - run the identity benchmark (ТЗ 15.6)"
	@echo "make measure-scene - time the cinema scene against its 2 s budget (ТЗ сценарий 1)"
	@echo "make measure-presence - time the greeting and the stranger alert (ТЗ 15.3)"
	@echo "make migrate       - apply pending SQLite migrations to data/hub.db"
	@echo "make skill name=X  - scaffold a new skill in skills/X"

hub:
	$(PY) -m hub.main --config $(CONFIG)

client:
	$(PY) -m client.main --config $(CONFIG)

test:
	$(PY) -m pytest $(TESTS) -q
	$(PY) -m ruff check .
	$(PY) -m mypy common

test-regress:
	$(PY) -m pytest tests/regress -q

test-identity:
	$(PY) -m scripts.identity_benchmark --manifest data/identity_benchmark/manifest.json

measure-scene:
	$(PY) -m scripts.measure_scene_latency

measure-presence:
	$(PY) -m scripts.measure_presence_latency

migrate:
	$(PY) -m hub.migrations_runner --db data/hub.db

skill:
	@test -n "$(name)" || (echo "usage: make skill name=<name>" && exit 2)
	$(PY) -m hub.skill_scaffold --name $(name)
