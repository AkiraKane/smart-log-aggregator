"""Log aggregation and categorization engine."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class LogLevel(str, Enum):
    """Standard log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class LogCategory:
    """Represents a category of aggregated log entries."""

    name: str
    level: LogLevel
    pattern: str
    count: int = 0
    sample_messages: list[str] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    sources: set[str] = field(default_factory=set)
    summary: str = ""

    def add_entry(self, entry: "LogEntry") -> None:
        """Add a log entry to this category."""
        self.count += 1
        if len(self.sample_messages) < 5:
            self.sample_messages.append(entry.message)
        if not self.first_seen or entry.timestamp < self.first_seen:
            self.first_seen = entry.timestamp
        if not self.last_seen or entry.timestamp > self.last_seen:
            self.last_seen = entry.timestamp
        if entry.source:
            self.sources.add(entry.source)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "level": self.level.value,
            "pattern": self.pattern,
            "count": self.count,
            "sample_messages": self.sample_messages,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "sources": sorted(self.sources),
            "summary": self.summary,
        }


@dataclass
class LogEntry:
    """Represents a single parsed log entry."""

    timestamp: str
    level: LogLevel
    message: str
    source: str = ""
    raw: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "message": self.message,
            "source": self.source,
            "metadata": self.metadata,
        }


# Common log format patterns
LOG_PATTERNS = [
    # JSON log format
    re.compile(r"^\s*\{.*\}\s*$"),
    # Syslog format: Jan  1 00:00:00 hostname program[pid]: message
    re.compile(
        r"^(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
        r"(?P<host>\S+)\s+(?P<program>\S+?)(?:\[(?P<pid>\d+)\])?:\s+(?P<message>.*)$"
    ),
    # ISO timestamp with level: 2024-01-01T00:00:00Z [INFO] message
    re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+"
        r"(?:\[(?P<level>\w+)\]|(?P<level2>\w+))\s+(?P<message>.*)$"
    ),
    # Common log format: 2024-01-01 00:00:00,000 - module - LEVEL - message
    re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,.]\d+)\s*[-]\s*"
        r"(?P<source>\S+)\s*[-]\s*(?P<level>\w+)\s*[-]\s*(?P<message>.*)$"
    ),
    # Simple: LEVEL: message or [LEVEL] message
    re.compile(
        r"^(?:\[(?P<level1>\w+)\]|(?P<level2>\w+):)\s*(?P<message>.*)$"
    ),
]

LEVEL_KEYWORDS: dict[str, LogLevel] = {
    "debug": LogLevel.DEBUG,
    "dbg": LogLevel.DEBUG,
    "info": LogLevel.INFO,
    "information": LogLevel.INFO,
    "warn": LogLevel.WARNING,
    "warning": LogLevel.WARNING,
    "err": LogLevel.ERROR,
    "error": LogLevel.ERROR,
    "critical": LogLevel.CRITICAL,
    "crit": LogLevel.CRITICAL,
    "fatal": LogLevel.CRITICAL,
    "panic": LogLevel.CRITICAL,
    "emergency": LogLevel.CRITICAL,
    "emerg": LogLevel.CRITICAL,
}

# Patterns for categorizing log messages
CATEGORY_PATTERNS: list[tuple[str, str, LogLevel]] = [
    ("connection_error", r"connection\s+(refused|reset|timeout|timed?\s*out|failed)", LogLevel.ERROR),
    ("out_of_memory", r"(out\s+of\s+memory|oom|cannot\s+allocate|memory\s+exhaust)", LogLevel.CRITICAL),
    ("permission_denied", r"(permission\s+denied|access\s+denied|forbidden|unauthorized)", LogLevel.ERROR),
    ("disk_full", r"(no\s+space\s+left|disk\s+full|filesystem\s+full)", LogLevel.CRITICAL),
    ("dns_error", r"(dns\s+(resolution|lookup)\s+failed|name\s+resolution\s+error)", LogLevel.ERROR),
    ("ssl_tls_error", r"(ssl|tls)\s+(error|handshake\s+fail|certificate)", LogLevel.ERROR),
    ("timeout", r"(timeout|timed?\s*out|deadline\s+exceeded)", LogLevel.WARNING),
    ("authentication_failure", r"(auth\w*\s+fail|invalid\s+(token|credential|password|api\s*key))", LogLevel.ERROR),
    ("database_error", r"(database|db|sql|query)\s+(error|fail|timeout|connection)", LogLevel.ERROR),
    ("http_error_5xx", r"(http|status)\s*(5\d{2}|5\d\d)\b", LogLevel.ERROR),
    ("http_error_4xx", r"(http|status)\s*(4\d{2}|4\d\d)\b", LogLevel.WARNING),
    ("startup", r"(start(ed|ing)?|initializ(ing|ed)|boot(ed|ing)?|listen(ing)?\s+on)", LogLevel.INFO),
    ("shutdown", r"(shut(ting)?\s*down|stop(ping|ped)?|terminat(ing|ed)|graceful\s+stop)", LogLevel.INFO),
    ("config_loaded", r"(config\w*\s+load|configuration\s+(loaded|parsed|applied))", LogLevel.INFO),
    ("health_check", r"(health\s*check|readiness|liveness|ready\s+probe)", LogLevel.INFO),
]


def detect_level(text: str) -> LogLevel:
    """Detect log level from text."""
    lower = text.lower().strip()
    # Check explicit level patterns first
    for keyword, level in LEVEL_KEYWORDS.items():
        # Match as a word boundary
        pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
        if pattern.search(lower):
            return level
    return LogLevel.UNKNOWN


def detect_timestamp(text: str) -> str:
    """Extract timestamp from log line."""
    # ISO format
    iso_match = re.search(
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?",
        text,
    )
    if iso_match:
        return iso_match.group()

    # Syslog format
    syslog_match = re.search(r"\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}", text)
    if syslog_match:
        return syslog_match.group()

    # Date-time with comma
    dt_match = re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,.]\d+", text)
    if dt_match:
        return dt_match.group()

    return ""


def parse_log_line(line: str, default_source: str = "") -> LogEntry:
    """Parse a single log line into a LogEntry."""
    stripped = line.strip()
    if not stripped:
        return LogEntry(
            timestamp="",
            level=LogLevel.UNKNOWN,
            message="",
            source=default_source,
            raw=line,
        )

    # Try JSON parsing first
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            return _parse_json_log(data, stripped, default_source)
        except json.JSONDecodeError:
            pass

    # Extract components
    timestamp = detect_timestamp(stripped)
    level = detect_level(stripped)
    source = default_source

    # Try to extract source/program from syslog format
    syslog_match = re.match(
        r"^\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+(\S+?)(?:\[\d+\])?:",
        stripped,
    )
    if syslog_match:
        source = syslog_match.group(1)

    # Try common format: timestamp - source - LEVEL - message
    common_match = re.match(
        r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,.]\d+\s*[-]\s*(\S+)",
        stripped,
    )
    if common_match and not source:
        source = common_match.group(1)

    return LogEntry(
        timestamp=timestamp,
        level=level,
        message=stripped,
        source=source,
        raw=line,
    )


def _parse_json_log(data: dict[str, Any], raw: str, default_source: str) -> LogEntry:
    """Parse a JSON-formatted log entry."""
    timestamp = (
        data.get("timestamp", "")
        or data.get("time", "")
        or data.get("@timestamp", "")
        or data.get("ts", "")
        or detect_timestamp(raw)
    )
    if isinstance(timestamp, (int, float)):
        timestamp = str(timestamp)

    level_str = (
        data.get("level", "")
        or data.get("severity", "")
        or data.get("loglevel", "")
        or data.get("lvl", "")
        or ""
    )
    level = LEVEL_KEYWORDS.get(str(level_str).lower(), detect_level(raw))

    message = (
        data.get("message", "")
        or data.get("msg", "")
        or data.get("text", "")
        or raw
    )

    source = (
        data.get("source", "")
        or data.get("logger", "")
        or data.get("service", "")
        or data.get("caller", "")
        or default_source
    )

    metadata = {k: v for k, v in data.items() if k not in {
        "timestamp", "time", "@timestamp", "ts",
        "level", "severity", "loglevel", "lvl",
        "message", "msg", "text",
        "source", "logger", "service", "caller",
    }}

    return LogEntry(
        timestamp=str(timestamp),
        level=level,
        message=str(message),
        source=str(source),
        raw=raw,
        metadata=metadata,
    )


def categorize_entry(entry: LogEntry) -> str:
    """Categorize a log entry based on its message content."""
    msg_lower = entry.message.lower()
    for cat_name, pattern, _ in CATEGORY_PATTERNS:
        if re.search(pattern, msg_lower):
            return cat_name
    return f"other_{entry.level.value}"


def aggregate_logs(
    entries: list[LogEntry],
    custom_patterns: list[tuple[str, str, LogLevel]] | None = None,
) -> dict[str, LogCategory]:
    """Aggregate log entries into categories.

    Args:
        entries: Parsed log entries to categorize.
        custom_patterns: Optional additional (name, regex, level) tuples.

    Returns:
        Dictionary mapping category name to LogCategory.
    """
    all_patterns = list(CATEGORY_PATTERNS)
    if custom_patterns:
        all_patterns.extend(custom_patterns)

    categories: dict[str, LogCategory] = {}

    for entry in entries:
        msg_lower = entry.message.lower()
        matched = False

        for cat_name, pattern, default_level in all_patterns:
            if re.search(pattern, msg_lower):
                if cat_name not in categories:
                    categories[cat_name] = LogCategory(
                        name=cat_name,
                        level=max(entry.level, default_level, key=lambda x: _level_sort_key(x)),
                        pattern=pattern,
                    )
                categories[cat_name].add_entry(entry)
                matched = True
                break

        if not matched:
            cat_name = f"other_{entry.level.value}"
            if cat_name not in categories:
                categories[cat_name] = LogCategory(
                    name=cat_name,
                    level=entry.level,
                    pattern="(uncategorized)",
                )
            categories[cat_name].add_entry(entry)

    return categories


def _level_sort_key(level: LogLevel) -> int:
    """Return numeric priority for log levels (higher = more severe)."""
    order = {
        LogLevel.DEBUG: 0,
        LogLevel.INFO: 1,
        LogLevel.WARNING: 2,
        LogLevel.ERROR: 3,
        LogLevel.CRITICAL: 4,
        LogLevel.UNKNOWN: -1,
    }
    return order.get(level, -1)


def parse_log_lines(lines: list[str], default_source: str = "") -> list[LogEntry]:
    """Parse multiple log lines."""
    entries = []
    for line in lines:
        if line.strip():
            entries.append(parse_log_line(line, default_source))
    return entries


def read_log_file(filepath: str) -> list[LogEntry]:
    """Read and parse a log file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return parse_log_lines(lines, default_source=filepath)
