FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements_frontend.txt requirements_frontend.txt
COPY pyproject.toml pyproject.toml
COPY src src/

RUN pip install -r requirements_frontend.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["streamlit", "src/proj/frontend.py", "--server.port", "$PORT" "--workers", "1"]
