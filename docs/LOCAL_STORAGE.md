# Local File Storage for Crawled Data

Spiderweb now supports saving crawled content to local files in addition to (or instead of) vector storage!

## Quick Start

### Save to Local Files

```bash
# Save crawled content to local directory
spiderweb crawl https://example.com --save-to ./crawled_data

# Choose format (markdown, html, json, or all)
spiderweb crawl https://example.com \
  --save-to ./crawled_data \
  --save-format markdown
```

### Save AND Ingest (Best of Both Worlds)

```bash
# Save locally for backup AND ingest to vector store
spiderweb crawl https://docs.example.com \
  --save-to ./backup \
  --ingest \
  --store qdrant://localhost:6333/docs
```

## Storage Formats

### 1. Markdown (`.md`)
- Clean, readable text
- Includes frontmatter with metadata
- Perfect for documentation

**Example output:**
```markdown
---
url: https://example.com
crawled_at: 2024-01-26T10:30:00
status_code: 200
success: true
links_found: 25
---

# Page Title

Content here...
```

### 2. HTML (`.html`)
- Raw HTML content
- Preserves original structure
- Good for archiving

### 3. JSON (`.json`)
- Complete structured data
- Includes all metadata
- Easy to process programmatically

**Example output:**
```json
{
  "url": "https://example.com",
  "crawled_at": "2024-01-26T10:30:00",
  "status_code": 200,
  "success": true,
  "markdown": "...",
  "metadata": {...},
  "links": [...]
}
```

### 4. All (default)
Saves all three formats for maximum flexibility!

## File Organization

Files are automatically organized with safe filenames:

```
crawled_data/
├── example_com_20240126_103000.md
├── example_com_20240126_103000.html
├── example_com_20240126_103000.json
├── docs_example_com_page1_20240126_103015.md
├── docs_example_com_page1_20240126_103015.html
├── docs_example_com_page1_20240126_103015.json
└── index.json  # Auto-generated index of all files
```

**Filename format:** `{sanitized_url}_{timestamp}.{ext}`

## Use Cases

### 1. Documentation Backup

```bash
# Crawl documentation and save locally
spiderweb crawl https://docs.python.org \
  --depth 3 \
  --max-pages 100 \
  --save-to ./python-docs-backup \
  --save-format markdown
```

**Why:** Keep offline copies of documentation

### 2. Data Collection for Analysis

```bash
# Crawl and save as JSON for processing
spiderweb crawl https://news-site.com \
  --depth 2 \
  --save-to ./news-data \
  --save-format json
```

**Why:** Process crawled data with custom scripts

### 3. Hybrid Storage (Recommended!)

```bash
# Save locally AND ingest to vector store
spiderweb crawl https://company.com/docs \
  --save-to ./docs-backup \
  --ingest \
  --store qdrant://localhost:6333/docs
```

**Why:** 
- Vector store for semantic search
- Local files for backup/archiving
- Best of both worlds!

### 4. Web Archive Creation

```bash
# Archive entire website
spiderweb crawl https://example.com \
  --depth 3 \
  --max-pages 500 \
  --save-to ./archives/example-com \
  --save-format all \
  --delay 2.0
```

**Why:** Create searchable local archives

## Python API

```python
from spiderweb import Spiderweb
from gluellm import GlueLLM

async with Spiderweb(llm_client=GlueLLM()) as web:
    # Basic save
    result = await web.crawl(
        "https://example.com",
        save_to="./crawled_data",
        save_format="markdown"
    )
    
    # Save and ingest
    result = await web.crawl(
        "https://example.com",
        save_to="./backup",
        save_format="all",
        ingest=True,
    )
```

### Direct Storage API

```python
from spiderweb.crawlers.storage import CrawlStorage
from spiderweb.crawlers.http import HttpCrawler

# Create storage
storage = CrawlStorage(output_dir="./my_crawls")

# Crawl
crawler = HttpCrawler()
result = await crawler.crawl("https://example.com")

# Save
saved_files = storage.save_crawl_result(
    result,
    format="all",
    include_metadata=True
)

# Create index
index_file = storage.create_index()
print(f"Saved {len(saved_files)} files, index at {index_file}")
```

## Docker Usage

```bash
# Save to mounted directory
docker-compose exec spiderweb spiderweb crawl https://example.com \
  --save-to /app/data/crawled

# Access saved files on host
ls -la ./data/crawled/

# Or use Makefile
make crawl URL=https://example.com SAVE=/app/data/crawled
```

## Advanced Features

### Structured Data Storage

When using `--extract` with a Pydantic schema, the extracted data is also saved:

```bash
spiderweb crawl https://store.com/product \
  --extract \
  --semantic-guide "Extract product information" \
  --save-to ./products \
  --save-format json
```

**Output includes:**
- `product_page_data.json` - Extracted structured data
- `product_page.json` - Full crawl result with extraction in metadata

### Index File

An `index.json` file is automatically created listing all saved files:

```json
{
  "created_at": "2024-01-26T10:30:00",
  "output_dir": "./crawled_data",
  "files": [
    {
      "name": "example_com_20240126_103000.md",
      "size": 15234,
      "modified": "2024-01-26T10:30:00"
    }
  ]
}
```

**Use it to:**
- Track what was crawled
- Check file sizes
- Build processing pipelines

## Comparison: Local vs Vector Storage

| Feature | Local Files | Vector Store | Both |
|---------|-------------|--------------|------|
| **Semantic search** | ❌ | ✅ | ✅ |
| **Full content** | ✅ | ❌ (chunks only) | ✅ |
| **Offline access** | ✅ | ❌ | ✅ |
| **Easy backup** | ✅ | ⚠️ Complex | ✅ |
| **Query speed** | ❌ | ✅ | ✅ |
| **Storage size** | Small | Large | Medium |
| **Processing** | Easy | Complex | Easy |

**Recommendation:** Use both! 
- Local files for backup and archiving
- Vector store for semantic search and RAG

## Tips & Best Practices

### 1. Organize by Date

```bash
mkdir -p ./crawls/$(date +%Y-%m-%d)
spiderweb crawl https://example.com --save-to ./crawls/$(date +%Y-%m-%d)
```

### 2. Use Markdown for Readability

```bash
# Save as markdown for easy viewing/editing
spiderweb crawl https://docs.example.com \
  --save-to ./docs \
  --save-format markdown
```

### 3. Use JSON for Processing

```bash
# Save as JSON for programmatic processing
spiderweb crawl https://api-docs.com \
  --save-to ./api-data \
  --save-format json
```

### 4. Combine with Git

```bash
# Save to git-tracked directory
spiderweb crawl https://company.com/docs --save-to ./docs-archive
cd docs-archive
git add .
git commit -m "Crawled docs on $(date)"
git push
```

**Benefits:** Version control for crawled content!

### 5. Process Saved Files

```python
import json
from pathlib import Path

# Process all saved JSON files
crawl_dir = Path("./crawled_data")
for json_file in crawl_dir.glob("*.json"):
    data = json.loads(json_file.read_text())
    
    # Extract information
    url = data["url"]
    links = data["links"]
    
    # Process...
    print(f"Found {len(links)} links on {url}")
```

## Troubleshooting

### Permission Denied

```bash
# Ensure directory is writable
chmod 755 ./crawled_data
spiderweb crawl https://example.com --save-to ./crawled_data
```

### Disk Space

Check available space before large crawls:

```bash
df -h
# Then crawl with limits
spiderweb crawl https://large-site.com \
  --max-pages 100 \
  --save-to ./data
```

### File Name Too Long

Files are automatically truncated to safe lengths (200 chars). If you hit issues:

```bash
# Use shorter output directory names
spiderweb crawl https://very-long-domain-name.com --save-to ./c
```

## Related Features

- [Crawl Feature Documentation](./CRAWL_FEATURE.md)
- Vector Storage Guide (not in this repo yet)
- [Docker Setup](../docker/README.md)

## Summary

Local file storage gives you:
- ✅ **Backups** - Keep original content
- ✅ **Offline access** - No database needed
- ✅ **Easy processing** - Standard file formats
- ✅ **Version control** - Track changes with git
- ✅ **Flexibility** - Choose your format

**Best practice:** Use `--save-to` AND `--ingest` together for the best of both worlds! 🎯

