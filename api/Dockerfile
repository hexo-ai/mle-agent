FROM python:3.11-slim as python-base

WORKDIR  /tmp

RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock* /tmp/

RUN poetry export -f requirements.txt --output requirements.txt --with test --without-hashes

FROM python:3.11-slim as production

WORKDIR /app

COPY --from=python-base /tmp/requirements.txt /app/requirements.txt

RUN apt-get update; apt-get install -y gcc g++ git
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./ .

EXPOSE 8000

RUN chmod 700 ./startup.sh
CMD ["./startup.sh"]
