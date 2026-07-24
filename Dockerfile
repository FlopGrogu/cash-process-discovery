# syntax=docker/dockerfile:1.7
FROM eclipse-temurin:8u462-b08-jre-jammy AS java8

FROM python:3.11.15-slim-bookworm

ARG UV_VERSION=0.8.13
ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:/opt/java/openjdk/bin:${PATH}" \
    DATA_ROOT=/data \
    RESULTS_ROOT=/results \
    LOG_ROOT=/logs/slurm \
    SPLIT_MINER_JAR=/inputs/split-miner-1.7.1-all.jar

COPY --from=java8 /opt/java/openjdk /opt/java/openjdk
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates gcc g++ git swig \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY environments/gedi/pyproject.toml environments/gedi/uv.lock environments/gedi/
RUN UV_PROJECT_ENVIRONMENT=/opt/gedi-venv \
    uv sync --project environments/gedi --frozen

COPY . .
RUN uv sync --frozen --all-extras \
    && rm -rf /app/environments/gedi/.venv \
    && ln -sfn /opt/gedi-venv environments/gedi/.venv \
    && chmod 0755 container/verify-split-miner.sh \
    && java -version \
    && environments/gedi/.venv/bin/python -c \
       "from importlib.metadata import version; print('GEDI', version('gedi'))"

CMD ["pdcash-verify-inputs", "--all", "--json"]
