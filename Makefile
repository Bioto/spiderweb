.PHONY: help init build up down restart logs shell crawl test clean

# Default target
help:
	@echo "Spiderweb Docker Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make init         - Create necessary directories"
	@echo "  make build        - Build Docker images"
	@echo "  make up           - Start all services"
	@echo "  make down         - Stop all services"
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Start in development mode (with code mounting)"
	@echo "  make shell        - Open shell in container"
	@echo "  make logs         - View logs"
	@echo "  make restart      - Restart services"
	@echo ""
	@echo "Usage:"
	@echo "  make crawl URL=https://example.com"
	@echo "  make query Q=\"search term\""
	@echo "  make ingest PATH=/app/examples"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run tests in container"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Stop and remove containers/volumes"
	@echo "  make clean-all    - Deep clean (including images)"

# Initialize directories
init:
	@echo "Creating necessary directories..."
	@mkdir -p logs data
	@chmod 755 logs data
	@echo "✓ Directories created"

# Build images
build: init
	docker-compose build

# Start services
up: init
	docker-compose up -d
	@echo ""
	@echo "✓ Services started!"
	@echo "  Qdrant Dashboard: http://localhost:6333/dashboard"
	@echo ""
	@echo "Run commands with:"
	@echo "  make crawl URL=https://example.com"

# Start in development mode
dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
	@echo ""
	@echo "✓ Development mode started with live code mounting"

# Stop services
down:
	docker-compose down

# Restart services
restart:
	docker-compose restart

# View logs
logs:
	docker-compose logs -f spiderweb

# Open shell in container
shell:
	docker-compose exec spiderweb /bin/bash

# Crawl a URL
crawl:
ifndef URL
	@echo "Error: URL not specified"
	@echo "Usage: make crawl URL=https://example.com"
	@echo "       make crawl URL=https://example.com SAVE=./data"
	@exit 1
endif
ifdef SAVE
	docker-compose exec spiderweb spiderweb crawl $(URL) --save-to $(SAVE) $(ARGS)
else
	docker-compose exec spiderweb spiderweb crawl $(URL) $(ARGS)
endif

# Query vector store
query:
ifndef Q
	@echo "Error: Query not specified"
	@echo "Usage: make query Q=\"search term\""
	@exit 1
endif
	docker-compose exec spiderweb spiderweb query "$(Q)" --store qdrant://qdrant:6333/docs $(ARGS)

# Ingest files
ingest:
ifndef PATH
	@echo "Error: PATH not specified"
	@echo "Usage: make ingest PATH=/app/examples"
	@exit 1
endif
	docker-compose exec spiderweb spiderweb ingest $(PATH) --store qdrant://qdrant:6333/docs $(ARGS)

# Run tests (in Docker)
test:
	docker-compose exec spiderweb pytest tests/ -v

# Run tests locally (without Docker)
test-local:
	uv run pytest tests/ -v

# Run tests with coverage
test-cov:
	uv run pytest tests/ -v --cov=spiderweb --cov-report=term-missing

# Clean up containers and volumes
clean:
	docker-compose down -v

# Deep clean (including images)
clean-all:
	docker-compose down -v --rmi all
	docker system prune -f

# Check if Qdrant is ready
check-qdrant:
	@echo "Checking Qdrant..."
	@curl -s http://localhost:6333/collections || echo "Qdrant not ready"

# Quick examples
example-basic:
	docker-compose exec spiderweb spiderweb crawl https://example.com

example-save-local:
	docker-compose exec spiderweb spiderweb crawl https://example.com --save-to /app/data/crawled

example-crawl-depth:
	docker-compose exec spiderweb spiderweb crawl https://example.com --depth 2 --max-pages 10

example-ingest:
	docker-compose exec spiderweb spiderweb crawl https://example.com \
		--ingest --store qdrant://qdrant:6333/docs

example-save-and-ingest:
	docker-compose exec spiderweb spiderweb crawl https://example.com \
		--save-to /app/data/backup --ingest --store qdrant://qdrant:6333/docs

example-query:
	docker-compose exec spiderweb spiderweb query "example" \
		--store qdrant://qdrant:6333/docs --top-k 5

