# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Corrected this changelog to describe **Spiderweb** (it previously contained GlueLLM-specific entries).

## [0.1.0]

### Added
- Core async `Spiderweb` API for **ingestion**, **querying**, and **web crawling**
- Multi-format extraction via `markitdown` (with optional `pdf`, `office`, and `ocr` extras)
- Pluggable chunking strategies: hierarchical, semantic, sliding-window, and sentence-based
- Quality gates via pluggable validators (e.g., deduplication and content-quality validation)
- Vector stores: **Qdrant** and an in-memory store for development
- Query expansion: **multi-query** and **HyDE** with RRF fusion
- Web crawling backends: simple HTTP and Crawl4AI (JS rendering), plus optional schema-guided extraction
- Local crawl-result storage to disk (markdown/html/json) with an auto-generated index
- CLI commands for crawling and querying; Docker setup for crawling + local Qdrant
