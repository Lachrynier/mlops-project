FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements_api.txt .
COPY pyproject.toml .
COPY configs configs/
COPY src src/

RUN pip install -r requirements_api.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["python",  "-u", "src/proj/api.py"]
