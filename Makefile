.PHONY: up down lint test render-config

up:
	docker compose -f deploy/compose/docker-compose.yml up

down:
	docker compose -f deploy/compose/docker-compose.yml down

lint:
	ruff check .
	ruff format --check .
	mypy common

test:
	pytest -q

render-config:
	python -m ftctl.cli render freqtrade --out freqtrade/user_data/config.json
