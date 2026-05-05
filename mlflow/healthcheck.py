import urllib.request
import sys
import os

# Get the port from environment or default to 5000
port = os.getenv("MLFLOW_SERVICE_PORT", "5000")
url = f"http://127.0.0.1:{port}/health"

try:
    # We set a short timeout so the healthcheck doesn't hang
    with urllib.request.urlopen(url, timeout=3) as response:
        if response.getcode() == 200:
            sys.exit(0)  # Healthy
        else:
            sys.exit(1)  # Unhealthy
except Exception:
    sys.exit(1)  # Unhealthy (Connection refused, etc.)
