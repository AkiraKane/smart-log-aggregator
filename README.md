# Smart Log Aggregator

**Day 21** of the DevOps + AI project series.

A Docker-based ELK alternative that uses AI to categorize and summarize logs. No Elasticsearch needed.

## Features

- Parse multiple log formats (JSON, syslog, ISO timestamps, common log format)
- Automatic log categorization (connection errors, OOM, timeouts, etc.)
- AI-powered log summarization using LLM (Ollama or OpenAI)
- Supports stdin streaming, single files, and directory scanning
- JSON and human-readable output modes
- Custom pattern support for domain-specific categorization
- Python stdlib only - no external dependencies

## Architecture

```
log-files --> aggregator.py (parse + categorize) --> LogCategory groups
                                                        |
                                                        v
                                                  llm.py (Ollama/OpenAI)
                                                        |
                                                        v
                                                  AI summaries per category
```

## Requirements

- Python 3.11+
- Ollama running at localhost:11434 (or OpenAI API key) for AI features
- No external Python dependencies (stdlib only)

## Quick Start

```bash
# Aggregate log file (no AI needed)
python -m src.main aggregate /var/log/syslog

# Aggregate with JSON output
python -m src.main aggregate /var/log/app.log --json

# Analyze with AI summaries
python -m src.main analyze /var/log/app.log

# Stream from stdin
cat /var/log/syslog | python -m src.main stream

# Run with Docker Compose
docker compose up
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
smart-log-aggregator/
  src/
    aggregator.py  - Log parsing and categorization engine (LogCategory, LogEntry)
    llm.py         - LLM client with Ollama/OpenAI fallback
    main.py        - CLI entry point
  tests/
    test_aggregator.py - Unit tests for aggregator
  Dockerfile
  docker-compose.yml
  .github/workflows/ci.yml
```

## Supported Log Formats

- **JSON**: `{"level": "error", "message": "..."}`
- **ISO timestamp**: `2024-01-15T10:30:00Z [ERROR] message`
- **Syslog**: `Jan 15 10:30:00 hostname app[pid]: message`
- **Common log**: `2024-01-15 10:30:00,000 - module - LEVEL - message`
- **Simple**: `ERROR: message`

## Built-in Categories

connection_error, out_of_memory, permission_denied, disk_full, dns_error, ssl_tls_error, timeout, authentication_failure, database_error, http_error_5xx, http_error_4xx, startup, shutdown, config_loaded, health_check
