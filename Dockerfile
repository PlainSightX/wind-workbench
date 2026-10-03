FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY infra/runtime-requirements.txt /tmp/runtime-requirements.txt
RUN pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir --require-hashes --only-binary=:all: -r /tmp/runtime-requirements.txt
COPY infra/torch-cpu-requirements.txt /tmp/torch-cpu-requirements.txt
RUN pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir --require-hashes --only-binary=:all: -r /tmp/torch-cpu-requirements.txt && pip check
COPY src /app/src
COPY infra/migrations /app/infra/migrations
COPY alembic.ini /app/alembic.ini
RUN groupadd --gid 10001 wind && useradd --uid 10001 --gid wind --no-create-home wind && mkdir /app/artifacts && chown wind:wind /app/artifacts
USER wind
CMD ["uvicorn", "power_forecast_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
