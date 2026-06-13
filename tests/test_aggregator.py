"""Tests for the aggregator module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.aggregator import (
    LogCategory,
    LogEntry,
    LogLevel,
    aggregate_logs,
    categorize_entry,
    detect_level,
    detect_timestamp,
    parse_log_line,
    parse_log_lines,
    read_log_file,
)


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_level_values(self) -> None:
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"


class TestDetectLevel:
    """Tests for log level detection."""

    def test_detect_info(self) -> None:
        assert detect_level("INFO: Application started") == LogLevel.INFO

    def test_detect_warning(self) -> None:
        assert detect_level("WARNING: Disk space low") == LogLevel.WARNING

    def test_detect_error(self) -> None:
        assert detect_level("ERROR: Connection failed") == LogLevel.ERROR

    def test_detect_critical(self) -> None:
        assert detect_level("CRITICAL: Out of memory") == LogLevel.CRITICAL

    def test_detect_debug(self) -> None:
        assert detect_level("DEBUG: Variable x = 42") == LogLevel.DEBUG

    def test_detect_warn_short(self) -> None:
        assert detect_level("WARN: Deprecated API") == LogLevel.WARNING

    @pytest.mark.parametrize("text,expected", [
        ("err: something broke", LogLevel.ERROR),
        ("The error occurred", LogLevel.ERROR),
        ("Fatal exception in thread", LogLevel.CRITICAL),
        ("Panic: nil pointer", LogLevel.CRITICAL),
        ("Service is starting", LogLevel.INFO),
        ("Application initialized", LogLevel.INFO),
        ("No level here", LogLevel.UNKNOWN),
    ])
    def test_detect_various_patterns(self, text: str, expected: LogLevel) -> None:
        assert detect_level(text) == expected

    def test_detect_unknown(self) -> None:
        assert detect_level("Just a plain message") == LogLevel.UNKNOWN

    def test_detect_case_insensitive(self) -> None:
        assert detect_level("error: failed") == LogLevel.ERROR
        assert detect_level("Error: failed") == LogLevel.ERROR
        assert detect_level("ERROR: failed") == LogLevel.ERROR

    def test_detect_in_brackets(self) -> None:
        assert detect_level("[WARNING] something") == LogLevel.WARNING


class TestDetectTimestamp:
    """Tests for timestamp detection."""

    def test_iso_format(self) -> None:
        ts = detect_timestamp("2024-01-15T10:30:00Z something happened")
        assert ts == "2024-01-15T10:30:00Z"

    def test_iso_with_offset(self) -> None:
        ts = detect_timestamp("2024-01-15T10:30:00+05:30 log entry")
        assert ts == "2024-01-15T10:30:00+05:30"

    def test_iso_with_space(self) -> None:
        ts = detect_timestamp("2024-01-15 10:30:00 something")
        assert ts == "2024-01-15 10:30:00"

    def test_syslog_format(self) -> None:
        ts = detect_timestamp("Jan 15 10:30:00 hostname app: message")
        assert ts == "Jan 15 10:30:00"

    def test_no_timestamp(self) -> None:
        assert detect_timestamp("just a message") == ""

    def test_iso_with_millis(self) -> None:
        ts = detect_timestamp("2024-01-15T10:30:00.123Z entry")
        assert ts == "2024-01-15T10:30:00.123Z"


class TestParseLogLine:
    """Tests for single log line parsing."""

    def test_json_log(self) -> None:
        line = json.dumps({
            "timestamp": "2024-01-15T10:30:00Z",
            "level": "error",
            "message": "Connection refused",
            "service": "api-gateway",
        })
        entry = parse_log_line(line)
        assert entry.level == LogLevel.ERROR
        assert entry.message == "Connection refused"
        assert entry.source == "api-gateway"
        assert entry.timestamp == "2024-01-15T10:30:00Z"

    def test_json_log_alt_keys(self) -> None:
        line = json.dumps({
            "time": "2024-01-15T10:30:00Z",
            "lvl": "warn",
            "msg": "High latency",
            "logger": "http",
        })
        entry = parse_log_line(line)
        assert entry.level == LogLevel.WARNING
        assert entry.message == "High latency"

    def test_iso_with_level(self) -> None:
        line = "2024-01-15T10:30:00Z [ERROR] Database connection failed"
        entry = parse_log_line(line)
        assert entry.level == LogLevel.ERROR
        assert entry.timestamp == "2024-01-15T10:30:00Z"

    def test_syslog_format(self) -> None:
        line = "Jan 15 10:30:00 webserver nginx[1234]: Connection timeout"
        entry = parse_log_line(line)
        assert entry.level == LogLevel.UNKNOWN  # no explicit level keyword in message
        assert entry.source == "nginx"
        assert entry.timestamp == "Jan 15 10:30:00"

    def test_simple_level_prefix(self) -> None:
        entry = parse_log_line("ERROR: Something went wrong")
        assert entry.level == LogLevel.ERROR

    def test_empty_line(self) -> None:
        entry = parse_log_line("")
        assert entry.message == ""
        assert entry.level == LogLevel.UNKNOWN

    def test_whitespace_only(self) -> None:
        entry = parse_log_line("   \n  ")
        assert entry.message == ""

    def test_default_source(self) -> None:
        entry = parse_log_line("INFO: started", default_source="myapp")
        assert entry.source == "myapp"

    def test_raw_preserved(self) -> None:
        line = "ERROR: test message"
        entry = parse_log_line(line)
        assert entry.raw == line

    def test_json_with_metadata(self) -> None:
        line = json.dumps({
            "level": "info",
            "message": "request completed",
            "method": "GET",
            "path": "/api/v1",
            "status": 200,
            "duration_ms": 42,
        })
        entry = parse_log_line(line)
        assert entry.metadata["method"] == "GET"
        assert entry.metadata["status"] == 200
        assert entry.metadata["duration_ms"] == 42

    def test_common_log_format(self) -> None:
        line = "2024-01-15 10:30:00,123 - myapp - INFO - Application started"
        entry = parse_log_line(line)
        assert entry.level == LogLevel.INFO
        assert entry.source == "myapp"
        assert "10:30:00" in entry.timestamp


class TestParseLogLines:
    """Tests for parsing multiple lines."""

    def test_parse_multiple_lines(self) -> None:
        lines = [
            "INFO: Starting application",
            "WARNING: Config value missing, using default",
            "ERROR: Database connection failed",
            "",
            "INFO: Retry successful",
        ]
        entries = parse_log_lines(lines)
        assert len(entries) == 4  # Empty line skipped

    def test_parse_with_source(self) -> None:
        lines = ["INFO: test1", "ERROR: test2"]
        entries = parse_log_lines(lines, default_source="app")
        assert all(e.source == "app" for e in entries)


class TestLogEntry:
    """Tests for LogEntry dataclass."""

    def test_to_dict(self) -> None:
        entry = LogEntry(
            timestamp="2024-01-15T10:30:00Z",
            level=LogLevel.ERROR,
            message="test",
            source="app",
        )
        d = entry.to_dict()
        assert d["level"] == "error"
        assert d["message"] == "test"


class TestLogCategory:
    """Tests for LogCategory dataclass."""

    def test_add_entry(self) -> None:
        cat = LogCategory(
            name="test_category",
            level=LogLevel.ERROR,
            pattern="test",
        )
        entry = LogEntry(
            timestamp="2024-01-15T10:30:00Z",
            level=LogLevel.ERROR,
            message="Error happened",
            source="app",
        )
        cat.add_entry(entry)
        assert cat.count == 1
        assert len(cat.sample_messages) == 1
        assert cat.first_seen == "2024-01-15T10:30:00Z"
        assert "app" in cat.sources

    def test_max_sample_messages(self) -> None:
        cat = LogCategory(
            name="test",
            level=LogLevel.INFO,
            pattern="test",
        )
        for i in range(10):
            entry = LogEntry(
                timestamp=f"2024-01-15T10:00:{i:02d}Z",
                level=LogLevel.INFO,
                message=f"Message {i}",
            )
            cat.add_entry(entry)
        assert len(cat.sample_messages) == 5
        assert cat.count == 10

    def test_timestamp_tracking(self) -> None:
        cat = LogCategory(name="test", level=LogLevel.INFO, pattern="test")
        cat.add_entry(LogEntry(timestamp="2024-01-15T10:00:00Z", level=LogLevel.INFO, message="a"))
        cat.add_entry(LogEntry(timestamp="2024-01-15T09:00:00Z", level=LogLevel.INFO, message="b"))
        cat.add_entry(LogEntry(timestamp="2024-01-15T11:00:00Z", level=LogLevel.INFO, message="c"))
        assert cat.first_seen == "2024-01-15T09:00:00Z"
        assert cat.last_seen == "2024-01-15T11:00:00Z"

    def test_to_dict(self) -> None:
        cat = LogCategory(
            name="connection_error",
            level=LogLevel.ERROR,
            pattern="connection.*failed",
            count=5,
        )
        d = cat.to_dict()
        assert d["name"] == "connection_error"
        assert d["count"] == 5


class TestCategorizeEntry:
    """Tests for entry categorization."""

    def test_connection_error(self) -> None:
        entry = LogEntry(timestamp="", level=LogLevel.ERROR, message="Connection refused to database")
        assert categorize_entry(entry) == "connection_error"

    def test_out_of_memory(self) -> None:
        entry = LogEntry(timestamp="", level=LogLevel.CRITICAL, message="Out of memory: cannot allocate 256MB")
        assert categorize_entry(entry) == "out_of_memory"

    def test_dns_error(self) -> None:
        entry = LogEntry(timestamp="", level=LogLevel.ERROR, message="DNS resolution failed for api.example.com")
        assert categorize_entry(entry) == "dns_error"

    def test_timeout(self) -> None:
        entry = LogEntry(timestamp="", level=LogLevel.WARNING, message="Request timeout after 30s")
        assert categorize_entry(entry) == "timeout"

    def test_http_5xx(self) -> None:
        entry = LogEntry(timestamp="", level=LogLevel.ERROR, message="HTTP 503 Service Unavailable")
        assert categorize_entry(entry) == "http_error_5xx"

    def test_startup(self) -> None:
        entry = LogEntry(timestamp="", level=LogLevel.INFO, message="Server started on port 8080")
        assert categorize_entry(entry) == "startup"

    def test_unknown_category(self) -> None:
        entry = LogEntry(timestamp="", level=LogLevel.INFO, message="Random log message")
        assert categorize_entry(entry) == "other_info"


class TestAggregateLogs:
    """Tests for log aggregation."""

    def test_aggregate_basic(self) -> None:
        entries = [
            LogEntry(timestamp="2024-01-15T10:00:00Z", level=LogLevel.ERROR,
                     message="Connection refused to db"),
            LogEntry(timestamp="2024-01-15T10:01:00Z", level=LogLevel.ERROR,
                     message="Connection refused to cache"),
            LogEntry(timestamp="2024-01-15T10:02:00Z", level=LogLevel.INFO,
                     message="Server started on port 8080"),
        ]
        categories = aggregate_logs(entries)
        assert "connection_error" in categories
        assert categories["connection_error"].count == 2
        assert "startup" in categories
        assert categories["startup"].count == 1

    def test_aggregate_empty(self) -> None:
        categories = aggregate_logs([])
        assert categories == {}

    def test_aggregate_with_custom_patterns(self) -> None:
        entries = [
            LogEntry(timestamp="", level=LogLevel.INFO,
                     message="User login successful for admin"),
        ]
        custom = [("user_login", r"user\s+login", LogLevel.INFO)]
        categories = aggregate_logs(entries, custom_patterns=custom)
        assert "user_login" in categories
        assert categories["user_login"].count == 1

    def test_aggregate_uncategorized(self) -> None:
        entries = [
            LogEntry(timestamp="", level=LogLevel.INFO,
                     message="Some random info message"),
        ]
        categories = aggregate_logs(entries)
        assert "other_info" in categories

    def test_aggregate_mixed(self) -> None:
        entries = [
            LogEntry(timestamp="", level=LogLevel.ERROR, message="Connection refused"),
            LogEntry(timestamp="", level=LogLevel.ERROR, message="Connection reset"),
            LogEntry(timestamp="", level=LogLevel.WARNING, message="Timeout waiting"),
            LogEntry(timestamp="", level=LogLevel.CRITICAL, message="Out of memory"),
            LogEntry(timestamp="", level=LogLevel.INFO, message="Server started"),
            LogEntry(timestamp="", level=LogLevel.INFO, message="Config loaded"),
        ]
        categories = aggregate_logs(entries)
        total = sum(c.count for c in categories.values())
        assert total == 6

    def test_aggregate_sources_tracked(self) -> None:
        entries = [
            LogEntry(timestamp="", level=LogLevel.ERROR,
                     message="Connection refused", source="api"),
            LogEntry(timestamp="", level=LogLevel.ERROR,
                     message="Connection reset", source="web"),
        ]
        categories = aggregate_logs(entries)
        cat = categories["connection_error"]
        assert "api" in cat.sources
        assert "web" in cat.sources


class TestReadLogFile:
    """Tests for reading log files."""

    def test_read_log_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        log_file.write_text(
            "INFO: Application started\n"
            "WARNING: Low memory\n"
            "ERROR: Connection failed\n"
        )
        entries = read_log_file(str(log_file))
        assert len(entries) == 3
        assert entries[0].level == LogLevel.INFO
        assert entries[1].level == LogLevel.WARNING
        assert entries[2].level == LogLevel.ERROR
        assert entries[0].source == str(log_file)

    def test_read_json_log_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "app.log"
        lines = [
            json.dumps({"level": "info", "message": "started"}),
            json.dumps({"level": "error", "message": "failed"}),
        ]
        log_file.write_text("\n".join(lines) + "\n")
        entries = read_log_file(str(log_file))
        assert len(entries) == 2

    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_log_file(str(tmp_path / "missing.log"))

    def test_read_empty_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "empty.log"
        log_file.write_text("")
        entries = read_log_file(str(log_file))
        assert entries == []
