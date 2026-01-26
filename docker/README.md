# Docker Setup for Spiderweb with Playwright

This directory contains Docker configuration for running Spiderweb with full Playwright support, including pre-installed browsers.

## Quick Start

### 1. Set up environment variables

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
ANTHROPIC_API_KEY=your_anthropic_key
SPIDERWEB_LOG_LEVEL=INFO
SPIDERWEB_EMBEDDING_MODEL=openai:text-embedding-3-small
```

### 2. Build and start services

```bash
# Build the image (includes Playwright browsers)
docker-compose build

# Start all services (Spiderweb + Qdrant)
docker-compose up -d
```

### 3. Use Spiderweb CLI

```bash
# Run CLI commands in the container
docker-compose exec spiderweb spiderweb --help

# Example: Crawl a website
docker-compose exec spiderweb spiderweb crawl https://example.com

# Example: Crawl with JavaScript rendering (uses Playwright)
docker-compose exec spiderweb spiderweb crawl https://example.com \
  --provider crawl4ai \
  --depth 2

# Example: Crawl and ingest to Qdrant
docker-compose exec spiderweb spiderweb crawl https://docs.example.com \
  --ingest \
  --store qdrant://qdrant:6333/docs \
  --depth 2

# Example: Query the vector store
docker-compose exec spiderweb spiderweb query \
  "What is the main topic?" \
  --store qdrant://qdrant:6333/docs
```

### 4. Run Python scripts

```bash
# Run a Python script
docker-compose exec spiderweb python examples/crawl_usage.py

# Interactive Python shell
docker-compose exec spiderweb python
```

## Services

### spiderweb
- Main application container
- Includes Python 3.12, all dependencies, and Playwright with Chromium
- CLI available via `docker-compose exec spiderweb spiderweb ...`
- Volumes:
  - `./logs` - Application logs
  - `./data` - Data directory
  - `./examples` - Example scripts

### qdrant
- Vector database for storing embeddings
- Web UI: http://localhost:6333/dashboard
- gRPC API: localhost:6334
- Persistent storage in Docker volume

## Playwright in Docker

The Dockerfile includes these key features:

1. **System dependencies** - All required libraries for Chromium
2. **Browser pre-installation** - `playwright install --with-deps chromium`
3. **Environment setup** - Proper browser paths configured

This means:
- ✅ No need to run `playwright install` manually
- ✅ Browsers are ready to use immediately
- ✅ Consistent environment across machines
- ✅ Works with crawl4ai out of the box

## Advanced Usage

### Custom entrypoint for long-running tasks

```bash
# Run a crawl job in the background
docker-compose run -d spiderweb spiderweb crawl https://docs.example.com \
  --depth 3 \
  --max-pages 100 \
  --ingest \
  --store qdrant://qdrant:6333/docs
```

### Access Qdrant directly

```bash
# Qdrant REST API
curl http://localhost:6333/collections

# Qdrant dashboard
open http://localhost:6333/dashboard
```

### Development mode with live code

```yaml
# Add to docker-compose.override.yml
version: '3.8'
services:
  spiderweb:
    volumes:
      - ./spiderweb:/app/spiderweb  # Mount source code
    command: ["tail", "-f", "/dev/null"]  # Keep running
```

Then:
```bash
docker-compose up -d
docker-compose exec spiderweb spiderweb crawl https://example.com
```

### Build options

```bash
# Build without cache
docker-compose build --no-cache

# Build specific service
docker-compose build spiderweb

# Pull latest base images
docker-compose pull
```

## Troubleshooting

### Playwright browser not found

If you see browser errors:

```bash
# Rebuild image to reinstall browsers
docker-compose build --no-cache spiderweb
docker-compose up -d
```

### Memory issues

Playwright/Chromium can use significant memory. Increase Docker memory:

```bash
# Docker Desktop: Settings → Resources → Memory (recommend 4GB+)
```

Or add to docker-compose.yml:

```yaml
services:
  spiderweb:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Permission issues

```bash
# Fix log/data directory permissions
sudo chown -R $USER:$USER logs/ data/
```

### Port conflicts

If ports 6333/6334 are in use:

```yaml
services:
  qdrant:
    ports:
      - "16333:6333"  # Use different host port
      - "16334:6334"
```

## Production Deployment

For production, consider:

1. **Use specific versions** - Replace `:latest` tags with versions
2. **Resource limits** - Add CPU/memory limits
3. **Health checks** - Add healthcheck configurations
4. **Secrets management** - Use Docker secrets instead of env vars
5. **Logging** - Configure log drivers
6. **Backups** - Regularly backup Qdrant volume

Example production additions:

```yaml
services:
  spiderweb:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          memory: 2G
    healthcheck:
      test: ["CMD", "python", "-c", "import spiderweb"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Cleaning Up

```bash
# Stop services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## Related Documentation

- [Main README](../README.md)
- [Crawl Feature](../docs/CRAWL_FEATURE.md)
- [Playwright Documentation](https://playwright.dev/python/docs/docker)
- [Qdrant Documentation](https://qdrant.tech/documentation/)

