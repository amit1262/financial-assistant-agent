FROM python:3.12.13-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Expose MLFlow tracking server port
EXPOSE 5000

# Start MLFlow server using uvx (no installation needed)
# CMD ["uvx", "mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--allowed-hosts", "mlflow_service,localhost,127.0.0.1"]
