.PHONY: up down init-db ingest-kb reembed test

up:
	docker compose up --build

down:
	docker compose down

init-db:
	python scripts/init_db.py

ingest-kb:
	python scripts/ingest_kb.py --kb-path data/kb_items.json

reembed:
	python scripts/reembed_incremental.py

test:
	pytest -q
