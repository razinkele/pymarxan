FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for numpy/scipy
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e ".[all]" && \
    apt-get purge -y gcc g++ && \
    apt-get autoremove -y

EXPOSE 8000

CMD ["shiny", "run", "src/pymarxan_app/app.py", "--host", "0.0.0.0", "--port", "8000"]
