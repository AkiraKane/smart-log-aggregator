"""CLI entry point for Smart Log Aggregator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.aggregator import (
    LogCategory,
    LogLevel,
    aggregate_logs,
    parse_log_lines,
    parse_log_line,
    read_log_file,
)
from src.llm import LLMClient, LLMError


SUMMARIZE_SYSTEM_PROMPT = """You are an expert SRE analyzing aggregated log data. Given a log category with sample messages, provide:
1. A brief summary of what's happening
2. Likely root cause
3. Recommended next steps
4. Severity assessment

Be concise but actionable. Output in Markdown."""


def summarize_category(category: LogCategory, client: LLMClient) -> str:
    """Use LLM to summarize a log category."""
    prompt_parts = [
        f"Category: {category.name}",
        f"Level: {category.level.value}",
        f"Count: {category.count}",
        f"First seen: {category.first_seen}",
        f"Last seen: {category.last_seen}",
        f"Sources: {', '.join(sorted(category.sources))}",
        "",
        "Sample messages:",
    ]
    for i, msg in enumerate(category.sample_messages, 1):
        prompt_parts.append(f"  {i}. {msg[:500]}")

    return client.generate("\n".join(prompt_parts), system=SUMMARIZE_SYSTEM_PROMPT)


def print_category_report(categories: dict[str, LogCategory], use_json: bool = False) -> None:
    """Print a report of aggregated categories."""
    if use_json:
        data = {name: cat.to_dict() for name, cat in categories.items()}
        print(json.dumps(data, indent=2, default=str))
        return

    # Sort by severity then count
    sorted_cats = sorted(
        categories.values(),
        key=lambda c: (-_level_order(c.level), -c.count),
    )

    print(f"\n{'='*60}")
    print(f" Log Aggregation Report ({len(sorted_cats)} categories)")
    print(f"{'='*60}\n")

    for cat in sorted_cats:
        level_tag = cat.level.value.upper()
        print(f"[{level_tag}] {cat.name}")
        print(f"  Count: {cat.count}")
        print(f"  Pattern: {cat.pattern}")
        if cat.first_seen:
            print(f"  First seen: {cat.first_seen}")
        if cat.last_seen:
            print(f"  Last seen: {cat.last_seen}")
        if cat.sources:
            print(f"  Sources: {', '.join(sorted(cat.sources))}")
        if cat.sample_messages:
            print("  Sample:")
            for msg in cat.sample_messages[:3]:
                print(f"    - {msg[:120]}{'...' if len(msg) > 120 else ''}")
        if cat.summary:
            print(f"  AI Summary: {cat.summary}")
        print()


def _level_order(level: LogLevel) -> int:
    """Return numeric order for sorting (higher = more severe)."""
    order = {
        LogLevel.CRITICAL: 0,
        LogLevel.ERROR: 1,
        LogLevel.WARNING: 2,
        LogLevel.INFO: 3,
        LogLevel.DEBUG: 4,
        LogLevel.UNKNOWN: 5,
    }
    return order.get(level, 5)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Smart Log Aggregator - AI-powered log analysis",
    )
    sub = parser.add_subparsers(dest="command")

    # aggregate command
    agg = sub.add_parser("aggregate", help="Aggregate and categorize log files")
    agg.add_argument("path", help="Path to a log file or directory")
    agg.add_argument("--json", action="store_true", help="Output as JSON")
    agg.add_argument("--source", default="", help="Default source name")

    # analyze command
    analyze = sub.add_parser("analyze", help="Aggregate + AI summarize log files")
    analyze.add_argument("path", help="Path to a log file or directory")
    analyze.add_argument("--json", action="store_true", help="Output as JSON")
    analyze.add_argument("--ollama-url", default=None, help="Ollama server URL")
    analyze.add_argument("--ollama-model", default=None, help="Ollama model name")
    analyze.add_argument("--openai-key", default=None, help="OpenAI API key for fallback")

    # stream command (reads from stdin)
    stream = sub.add_parser("stream", help="Aggregate logs from stdin")
    stream.add_argument("--source", default="stdin", help="Source name")
    stream.add_argument("--json", action="store_true", help="Output as JSON")

    return parser


def read_logs_from_path(path_str: str, source: str = "") -> list:
    """Read logs from a file or directory."""
    path = Path(path_str)
    if path.is_file():
        return read_log_file(str(path))
    elif path.is_dir():
        entries = []
        for log_file in sorted(path.rglob("*.log")):
            entries.extend(read_log_file(str(log_file)))
        for log_file in sorted(path.rglob("*.txt")):
            entries.extend(read_log_file(str(log_file)))
        return entries
    else:
        print(f"Error: Path not found: {path}", file=sys.stderr)
        sys.exit(1)


def cmd_aggregate(args: argparse.Namespace) -> int:
    """Handle the aggregate command."""
    entries = read_logs_from_path(args.path, args.source)
    if not entries:
        print("No log entries found.", file=sys.stderr)
        return 1

    print(f"Parsed {len(entries)} log entries.")
    categories = aggregate_logs(entries)
    print_category_report(categories, use_json=args.json)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Handle the analyze command."""
    entries = read_logs_from_path(args.path)
    if not entries:
        print("No log entries found.", file=sys.stderr)
        return 1

    print(f"Parsed {len(entries)} log entries.")
    categories = aggregate_logs(entries)

    client = LLMClient(
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        openai_api_key=args.openai_key,
    )

    print(f"Generating AI summaries for {len(categories)} categories...")
    for name, cat in categories.items():
        try:
            cat.summary = summarize_category(cat, client)
        except LLMError as e:
            cat.summary = f"(LLM error: {e})"

    print_category_report(categories, use_json=args.json)
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    """Handle the stream command - read from stdin."""
    lines = sys.stdin.readlines()
    entries = parse_log_lines(lines, default_source=args.source)
    if not entries:
        print("No log entries found in stdin.", file=sys.stderr)
        return 1

    print(f"Parsed {len(entries)} log entries from stdin.", file=sys.stderr)
    categories = aggregate_logs(entries)
    print_category_report(categories, use_json=args.json)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "aggregate":
        return cmd_aggregate(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "stream":
        return cmd_stream(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
