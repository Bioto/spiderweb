FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for Playwright and other tools
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libwayland-client0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    libu2f-udev \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Copy source code (needed for installation)
COPY spiderweb/ ./spiderweb/
COPY README.md LICENSE ./

# Install Python dependencies including spiderweb itself
RUN uv pip install --system -e ".[all]"

# Install Playwright and its browsers AFTER Python packages are installed
# This is the key step that pre-installs browsers in the Docker image
RUN playwright install --with-deps chromium

# Verify playwright installation
RUN playwright --version && ls -la /ms-playwright/ || true

# Copy examples
COPY examples/ ./examples/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SPIDERWEB_LOG_LEVEL=INFO
# Playwright browsers installed in root's cache
ENV PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright

# Create directories
RUN mkdir -p /app/logs /app/data

# Default command
CMD ["spiderweb", "--help"]
