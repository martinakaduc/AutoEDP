"""
Utility script for analyzing vLLM middleware logs.

This script provides tools to analyze, visualize, and export logged
queries and responses from the vLLM server.
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from middleware import load_logs, QueryRecord


def analyze_logs(
    log_dir: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict:
    """
    Analyze logs and generate comprehensive statistics.

    Args:
        log_dir: Directory containing log files
        start_date: Optional start date filter (YYYYMMDD)
        end_date: Optional end date filter (YYYYMMDD)

    Returns:
        Dictionary containing analysis results
    """
    records = load_logs(log_dir, start_date, end_date)

    if not records:
        return {"error": "No records found"}

    # Basic statistics
    total = len(records)
    successful = [r for r in records if r.error is None]
    failed = [r for r in records if r.error is not None]

    # Latency statistics
    latencies = [r.latency_ms for r in successful if r.latency_ms is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0

    # Model usage
    model_counts = defaultdict(int)
    for record in records:
        model_counts[record.model] += 1

    # Error analysis
    error_types = defaultdict(int)
    for record in failed:
        if record.error:
            # Extract error type
            error_type = (
                record.error.split(":")[0] if ":" in record.error else record.error
            )
            error_types[error_type] += 1

    # Temperature distribution
    temps = [r.parameters.get("temperature", 0) for r in records]
    avg_temp = sum(temps) / len(temps) if temps else 0

    # Token usage (if available)
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0

    for record in successful:
        if record.response and "usage" in record.response:
            usage = record.response["usage"]
            total_tokens += usage.get("total_tokens", 0)
            prompt_tokens += usage.get("prompt_tokens", 0)
            completion_tokens += usage.get("completion_tokens", 0)

    # Time range
    timestamps = [datetime.fromisoformat(r.timestamp) for r in records]
    first_request = min(timestamps)
    last_request = max(timestamps)
    duration = (last_request - first_request).total_seconds()

    # Requests per hour
    requests_per_hour = (total / duration * 3600) if duration > 0 else 0

    return {
        "summary": {
            "total_requests": total,
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / total * 100 if total > 0 else 0,
        },
        "latency": {
            "average_ms": avg_latency,
            "min_ms": min_latency,
            "max_ms": max_latency,
        },
        "models": dict(model_counts),
        "errors": dict(error_types),
        "parameters": {
            "average_temperature": avg_temp,
        },
        "tokens": {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "avg_tokens_per_request": (
                total_tokens / len(successful) if successful else 0
            ),
        },
        "time_range": {
            "first_request": first_request.isoformat(),
            "last_request": last_request.isoformat(),
            "duration_seconds": duration,
            "requests_per_hour": requests_per_hour,
        },
    }


def export_for_training(
    log_dir: str,
    output_file: str,
    min_tokens: int = 10,
    max_tokens: int = 2048,
    exclude_errors: bool = True,
) -> int:
    """
    Export logs to a format suitable for training.

    Args:
        log_dir: Directory containing log files
        output_file: Output file path (JSONL format)
        min_tokens: Minimum number of tokens to include
        max_tokens: Maximum number of tokens to include
        exclude_errors: Whether to exclude failed requests

    Returns:
        Number of exported records
    """
    records = load_logs(log_dir)
    exported = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            # Filter based on criteria
            if exclude_errors and record.error is not None:
                continue

            if not record.response:
                continue

            # Check token count if usage info is available
            if "usage" in record.response:
                total_tokens = record.response["usage"].get("total_tokens", 0)
                if total_tokens < min_tokens or total_tokens > max_tokens:
                    continue

            # Extract the completion
            if "choices" in record.response and record.response["choices"]:
                choice = record.response["choices"][0]

                # Format for training
                training_example = {
                    "messages": record.messages,
                    "completion": choice.get("message", {}).get("content")
                    or choice.get("text"),
                    "model": record.model,
                    "parameters": record.parameters,
                }

                f.write(json.dumps(training_example) + "\n")
                exported += 1

    return exported


def find_slow_requests(
    log_dir: str,
    threshold_ms: float = 1000.0,
    output_file: Optional[str] = None,
) -> List[QueryRecord]:
    """
    Find requests that took longer than the threshold.

    Args:
        log_dir: Directory containing log files
        threshold_ms: Latency threshold in milliseconds
        output_file: Optional file to save results

    Returns:
        List of slow requests
    """
    records = load_logs(log_dir)
    slow_requests = [
        r for r in records if r.latency_ms is not None and r.latency_ms > threshold_ms
    ]

    # Sort by latency
    slow_requests.sort(key=lambda r: r.latency_ms or 0, reverse=True)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            for record in slow_requests:
                f.write(record.model_dump_json() + "\n")

    return slow_requests


def compare_models(log_dir: str) -> Dict:
    """
    Compare performance metrics across different models.

    Args:
        log_dir: Directory containing log files

    Returns:
        Dictionary with comparison results
    """
    records = load_logs(log_dir)
    model_stats = defaultdict(
        lambda: {
            "count": 0,
            "latencies": [],
            "errors": 0,
            "total_tokens": 0,
        }
    )

    for record in records:
        stats = model_stats[record.model]
        stats["count"] += 1

        if record.error:
            stats["errors"] += 1
        else:
            if record.latency_ms is not None:
                stats["latencies"].append(record.latency_ms)

            if record.response and "usage" in record.response:
                stats["total_tokens"] += record.response["usage"].get("total_tokens", 0)

    # Calculate averages
    comparison = {}
    for model, stats in model_stats.items():
        comparison[model] = {
            "total_requests": stats["count"],
            "error_rate": (
                stats["errors"] / stats["count"] * 100 if stats["count"] > 0 else 0
            ),
            "avg_latency_ms": (
                sum(stats["latencies"]) / len(stats["latencies"])
                if stats["latencies"]
                else 0
            ),
            "total_tokens": stats["total_tokens"],
            "avg_tokens_per_request": (
                stats["total_tokens"] / stats["count"] if stats["count"] > 0 else 0
            ),
        }

    return comparison


def main():
    """Command-line interface for log analysis."""
    parser = argparse.ArgumentParser(description="Analyze vLLM middleware logs")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze logs and show statistics"
    )
    analyze_parser.add_argument(
        "--log-dir", default="./logs/vllm", help="Log directory"
    )
    analyze_parser.add_argument("--start-date", help="Start date (YYYYMMDD)")
    analyze_parser.add_argument("--end-date", help="End date (YYYYMMDD)")
    analyze_parser.add_argument("--output", help="Output file for results (JSON)")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export logs for training")
    export_parser.add_argument("--log-dir", default="./logs/vllm", help="Log directory")
    export_parser.add_argument("--output", required=True, help="Output file (JSONL)")
    export_parser.add_argument(
        "--min-tokens", type=int, default=10, help="Minimum tokens"
    )
    export_parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Maximum tokens"
    )
    export_parser.add_argument(
        "--include-errors", action="store_true", help="Include failed requests"
    )

    # Slow requests command
    slow_parser = subparsers.add_parser("slow", help="Find slow requests")
    slow_parser.add_argument("--log-dir", default="./logs/vllm", help="Log directory")
    slow_parser.add_argument(
        "--threshold", type=float, default=1000.0, help="Latency threshold (ms)"
    )
    slow_parser.add_argument("--output", help="Output file")

    # Compare models command
    compare_parser = subparsers.add_parser("compare", help="Compare model performance")
    compare_parser.add_argument(
        "--log-dir", default="./logs/vllm", help="Log directory"
    )
    compare_parser.add_argument("--output", help="Output file (JSON)")

    args = parser.parse_args()

    if args.command == "analyze":
        print("Analyzing logs...")
        results = analyze_logs(args.log_dir, args.start_date, args.end_date)

        if "error" in results:
            print(f"Error: {results['error']}")
            return

        # Pretty print results
        print("\n" + "=" * 60)
        print("LOG ANALYSIS RESULTS")
        print("=" * 60)

        for section, data in results.items():
            print(f"\n{section.upper()}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  {data}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    elif args.command == "export":
        print("Exporting logs for training...")
        count = export_for_training(
            args.log_dir,
            args.output,
            args.min_tokens,
            args.max_tokens,
            not args.include_errors,
        )
        print(f"Exported {count} records to {args.output}")

    elif args.command == "slow":
        print(f"Finding requests slower than {args.threshold}ms...")
        slow = find_slow_requests(args.log_dir, args.threshold, args.output)

        print(f"\nFound {len(slow)} slow requests:")
        for i, record in enumerate(slow[:10], 1):
            print(f"{i}. {record.request_id}: {record.latency_ms:.2f}ms")
            print(f"   Model: {record.model}")
            print(f"   Messages: {len(record.messages)} messages")

        if len(slow) > 10:
            print(f"... and {len(slow) - 10} more")

        if args.output:
            print(f"\nAll slow requests saved to {args.output}")

    elif args.command == "compare":
        print("Comparing model performance...")
        comparison = compare_models(args.log_dir)

        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        for model, stats in comparison.items():
            print(f"\n{model}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(comparison, f, indent=2)
            print(f"\nComparison saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
