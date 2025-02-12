FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY pyproject.toml .
<<<<<<< HEAD
COPY src/ src/
COPY configs/ configs/
=======
COPY src src/
COPY configs configs/
COPY data/processed/ data/processed/

>>>>>>> main

WORKDIR /
# RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["sh", "-c", "python -u src/proj/data.py --raw-dir=/gcs/data_bucket_77/data/raw && python -u src/proj/train.py"]
