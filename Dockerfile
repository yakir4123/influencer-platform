# Use a lightweight Python base image
FROM python:3.12-slim-bookworm AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy pyproject.toml and uv.lock (if it exists)
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv sync (excluding development packages)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Final stage
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source code
COPY app /app/app

# Ensure standard python output behavior
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Add virtual environment bin directory to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose port and run the server
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
