FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, configs, and scripts
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY app/ app/

# Copy documentation and entrypoint
COPY docs/ docs/
COPY README.md .
COPY Docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000 8501

ENTRYPOINT ["/entrypoint.sh"]