FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scirpt.py .
COPY server.py .

# Railway injects PORT; default 8000
EXPOSE 8000

# HTTP transport for Railway
ENV MCP_TRANSPORT=http

CMD ["python", "server.py"]
