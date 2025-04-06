FROM docker.io/intel/oneapi-basekit:2025.0.2-0-devel-ubuntu24.04

RUN apt update && apt -y install python3 python3-pip libmagic1 git build-essential libgl1

RUN mkdir /app && mkdir /app/src && mkdir /app/models && mkdir /app/working
WORKDIR /app/working

COPY src /app/src/
COPY pyproject.toml uv.lock .python-version README.md LICENSE docker/server.yml /app/

RUN cd /app && python3 -m pip install --break-system-packages -U uv && \
    uv venv && \
    uv pip install setuptools torch && \
    uv sync --frozen && \
    uv cache clean

VOLUME /app/models
VOLUME /app/working
EXPOSE 8080

CMD uv run --no-sync azarrot --config /app/server.yml
