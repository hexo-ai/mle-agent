.PHONY: install
install:
	poetry install

.PHONY: run
run:
	poetry run uvicorn mle_agent.main:app --reload

.PHONY: test
test:
	poetry run pytest

.PHONY: build
build:
	docker-compose -f docker-compose.test.yml build

.PHONE: up
up: build
	docker-compose -f docker-compose.test.yml up
