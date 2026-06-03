#!/usr/bin/env python3
"""ARIA — Adaptive Research Intelligence Architecture.

Usage:
    python main.py "Your research query"
    python main.py "Your query" --output output/report.md
    python main.py "Your query" --state-file state/checkpoint.json
    python main.py "Your query" --stream
    python main.py "Your query" --quality-threshold 7.0
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Configure logging before imports that may log at module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aria.main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ARIA: Adaptive Research Intelligence Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", help="Research query to investigate")
    parser.add_argument(
        "--output",
        default="output/report.md",
        help="Output path for the research report (default: output/report.md)",
    )
    parser.add_argument(
        "--state-file",
        default="state/checkpoint.json",
        help="Path to checkpoint file for resuming interrupted runs",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=None,
        help="Minimum aggregate quality score to accept (0-10, default: from config)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream sections to stdout as they complete",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh even if a checkpoint exists",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    logging.getLogger().setLevel(args.log_level)

    # Lazy imports after logging is configured
    from aria.agents.orchestrator import OrchestratorAgent
    from aria.config import get_settings
    from aria.memory.state import SharedWorkingMemory
    from aria.report.generator import ReportGenerator

    settings = get_settings()

    # Override threshold if provided
    if args.quality_threshold is not None:
        settings = settings.model_copy(
            update={"quality_threshold": args.quality_threshold}
        )

    # Validate that at least one provider is configured
    try:
        from aria.providers.factory import create_provider
        provider = create_provider(settings)
        logger.info("LLM provider: %s", provider.provider_name())
    except ValueError as exc:
        logger.error("LLM provider configuration error: %s", exc)
        return 1

    state_file = None if args.no_resume else args.state_file
    Path("state").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

    memory = SharedWorkingMemory()
    orchestrator = OrchestratorAgent(
        settings=settings,
        memory=memory,
        stream=args.stream,
    )

    if args.stream:
        def on_section(section):
            print(f"\n{'='*60}")
            print(f"SECTION COMPLETE: {section.section_title}")
            print(f"{'='*60}")
            print(section.content[:300], "...")
        orchestrator.on_section_complete(on_section)

    print(f"\nARIA Research System")
    print(f"Provider: {provider.provider_name()}")
    print(f"Query: {args.query}")
    print(f"Quality threshold: {settings.quality_threshold}")
    print(f"Max replan cycles: {settings.max_replan_cycles}")
    print("-" * 60)

    start = time.time()
    try:
        sections = await orchestrator.run(
            query=args.query,
            state_file=state_file,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted — checkpoint saved to %s", state_file)
        await memory.persist(state_file or "state/checkpoint.json")
        return 130

    if not sections:
        logger.error("No sections were produced.")
        return 1

    # Generate report
    report_gen = ReportGenerator(settings=settings)
    report = await report_gen.run(
        query=args.query,
        sections=sections,
        memory=memory,
        output_path=args.output,
    )

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Research complete in {elapsed:.1f}s")
    print(f"Sections: {len(sections)}")
    print(f"Report: {args.output}")
    print(f"{'='*60}\n")

    # Print executive summary to stdout
    lines = report.split("\n")
    in_summary = False
    for line in lines:
        if line.startswith("## Executive Summary"):
            in_summary = True
            continue
        if in_summary and line.startswith("## "):
            break
        if in_summary:
            print(line)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
