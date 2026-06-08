# main.py
# CLI entry point for the AI Image Detector (Stage 1).

import os
import sys
import json
import argparse

from dotenv import load_dotenv
load_dotenv()

import pipeline


def _print_summary(result: dict) -> None:
    score   = result.get("stage1_ai_score", 0.0)
    verdict = result.get("stage1_verdict", "unknown")
    print(f"Image:    {result.get('image_path')}")
    print(f"Score:    {score:.4f}")
    print(f"Verdict:  {verdict}")
    print(f"Time:     {result.get('processing_time_ms')}ms")

    # Key signals
    sigs = result.get("stage1_signals", [])
    if sigs:
        print("Signals:")
        for s in sigs:
            icon = "🔴" if s["direction"] == "ai" else "🟢"
            print(f"  {icon} {s['signal']} (w={s['weight']}) — {s['detail'][:70]}")
    else:
        print("Signals:  (none)")


def _print_verbose(result: dict) -> None:
    _print_summary(result)
    print()
    print("Pipeline log:")
    for line in result.get("pipeline_log", []):
        print(f"  {line}")
    print()
    print("next_stage_input (ready for Stage 2):")
    print(json.dumps(result.get("next_stage_input", {}), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Image Detector — Stage 1 metadata analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py photo.jpg
  python main.py photo.jpg --verbose
  python main.py photo.jpg --json
        """,
    )
    parser.add_argument("image_path", help="Path to the image to analyze")
    parser.add_argument("--json",    action="store_true", help="Print full JSON result")
    parser.add_argument("--verbose", action="store_true", help="Summary + pipeline log + next_stage_input")

    args = parser.parse_args()

    try:
        result = pipeline.run(args.image_path)
    except Exception as exc:
        print(f"ERROR: pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    elif args.verbose:
        _print_verbose(result)
    else:
        _print_summary(result)


if __name__ == "__main__":
    main()
