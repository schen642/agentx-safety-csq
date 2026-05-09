FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN adduser --disabled-password agent
USER agent
WORKDIR /home/agent

COPY --chown=agent pyproject.toml README.md ./
COPY --chown=agent src src

RUN pip install --no-cache-dir --user \
    "a2a-sdk[http-server]>=0.2,<1.0" \
    "uvicorn[standard]>=0.20" \
    "litellm>=1.0"

ENV PATH="/home/agent/.local/bin:${PATH}"

ENTRYPOINT ["python", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
