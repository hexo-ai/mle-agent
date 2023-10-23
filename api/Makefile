.PHONY: install
install:
	poetry install

.PHONY: run
run:
	poetry run uvicorn mle_agent.main:app --reload

.PHONY: test
test:
	poetry run pytest
