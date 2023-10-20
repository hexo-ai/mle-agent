.PHONY: install
install:
	poetry install

.PHONY: run
run:
	poetry run uvicorn mle_agent.main:app --reload
