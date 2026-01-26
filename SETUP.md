# Setup Guide

Choose your preferred setup method:

## Option 1: Docker (Recommended - Zero Configuration)

**Best for:** Production, consistent environments, no dependency hassles

```bash
# 1. Create .env file
cp env.example .env
# Edit .env with your API keys (OPENAI_API_KEY, etc.)

# 2. Start services (includes Playwright browsers)
make build
make up

# 3. Use commands
make crawl URL=https://example.com
make query Q="search term"
```

**Advantages:**
- ✅ Playwright browsers pre-installed
- ✅ No system dependencies needed
- ✅ Includes Qdrant vector database
- ✅ Consistent across all machines
- ✅ Easy cleanup

See [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) for details.

---

## Option 2: Local Installation

**Best for:** Development, testing, integration with local tools

### Prerequisites

- Python 3.12+
- Node.js (for Playwright)

### Installation Steps

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install spiderweb
pip install -e ".[all]"

# 3. Install Playwright browsers (IMPORTANT!)
playwright install chromium

# 4. Set environment variables
export OPENAI_API_KEY="your-key-here"

# 5. Optional: Start Qdrant locally (for vector storage)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

### Usage

```bash
# Crawl a website
spiderweb crawl https://example.com

# Crawl with JavaScript rendering (uses Playwright)
spiderweb crawl https://example.com --provider crawl4ai

# Crawl and ingest
spiderweb crawl https://docs.example.com \
  --ingest \
  --store qdrant://localhost:6333/docs \
  --depth 2

# Query
spiderweb query "search term" --store qdrant://localhost:6333/docs
```

---

## Troubleshooting

### Playwright Browser Not Found

**Local installation:**
```bash
playwright install chromium
```

**Docker:**
Browsers are pre-installed. If you see errors, rebuild:
```bash
make clean
make build
make up
```

### Port Conflicts (Qdrant)

If port 6333 is in use:

**Local:** Use a different Qdrant port
```bash
docker run -p 16333:6333 qdrant/qdrant:latest
```

**Docker:** Edit `docker-compose.yml` ports section

### Permission Errors (Docker)

```bash
make init  # Creates directories with correct permissions
```

### Missing API Keys

Create `.env` file in project root:
```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here  # Optional
```

---

## Which Setup Should I Use?

| Feature | Docker | Local |
|---------|--------|-------|
| **Easy setup** | ✅ One command | ⚠️ Multiple steps |
| **Playwright** | ✅ Pre-installed | ⚠️ Manual install |
| **Qdrant** | ✅ Included | ⚠️ Separate container |
| **Portability** | ✅ Works anywhere | ⚠️ Depends on system |
| **Development** | ⚠️ Need rebuild | ✅ Immediate changes |
| **IDE integration** | ⚠️ Remote debugging | ✅ Native |
| **Performance** | ⚠️ Container overhead | ✅ Native speed |

**Recommendation:**
- **Use Docker** for production, demos, and consistent environments
- **Use Local** for active development and debugging

---

## Quick Command Reference

### Docker
```bash
make build                    # Build images
make up                       # Start services
make crawl URL=<url>         # Crawl a URL
make query Q="<query>"       # Query vector store
make shell                    # Open container shell
make logs                     # View logs
make down                     # Stop services
make clean                    # Remove everything
```

### Local
```bash
spiderweb crawl <url>                    # Crawl
spiderweb ingest <path>                  # Ingest files
spiderweb query "<query>" --store <url>  # Query
spiderweb --help                         # All commands
```

---

## Next Steps

1. ✅ Choose your setup method above
2. 📖 Read the [Crawl Feature Documentation](docs/CRAWL_FEATURE.md)
3. 🔍 Try the [Examples](examples/crawl_usage.py)
4. 🚀 Build something awesome!

