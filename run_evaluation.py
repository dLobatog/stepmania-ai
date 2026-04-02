#!/usr/bin/env python3
"""Generate validation charts for one or more audio clips and print summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from stepmania_ai.generate import generate_chart
from stepmania_ai.utils.sm_parser import parse_sm


def default_clips() -> list[Path]:
    clips = [
        Path("~/Downloads/migos_15s.mp3").expanduser(),
        Path("~/Downloads/migos_1min.mp3").expanduser(),
    ]
    return [clip for clip in clips if clip.exists()]


def chart_pattern_stats(chart) -> dict[str, float]:
    rows = chart.taps_only
    jumps = sum(1 for row in rows if len(row.tap_columns) >= 2)
    jacks = 0

    for prev, curr in zip(rows, rows[1:]):
        if (
            len(prev.tap_columns) == 1
            and len(curr.tap_columns) == 1
            and prev.tap_columns[0] == curr.tap_columns[0]
            and (curr.time - prev.time) < 0.40
        ):
            jacks += 1

    nps_values = [count for _, count in chart.nps_series]
    return {
        "notes": float(len(rows)),
        "jumps": float(jumps),
        "jump_rate": jumps / max(len(rows), 1),
        "jacks": float(jacks),
        "peak_nps": float(max(nps_values) if nps_values else 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on fixed audio clips")
    parser.add_argument("--onset-model", required=True)
    parser.add_argument("--pattern-model", required=True)
    parser.add_argument("--output-dir", default="eval")
    parser.add_argument(
        "--decode-strategy",
        choices=["ergonomic", "raw"],
        default="ergonomic",
    )
    parser.add_argument("clips", nargs="*", help="Audio clips to evaluate")
    args = parser.parse_args()

    clips = [Path(c).expanduser() for c in args.clips] if args.clips else default_clips()
    if not clips:
        raise SystemExit("No evaluation clips found. Pass clip paths explicitly.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for clip in clips:
        print(f"\n=== Evaluating {clip.name} ===")
        output_path = output_dir / f"{clip.stem}.sm"
        generate_chart(
            audio_path=clip,
            onset_model_path=args.onset_model,
            pattern_model_path=args.pattern_model,
            output_path=output_path,
            title=clip.stem,
            decode_strategy=args.decode_strategy,
            difficulty="Challenge",
            rating=10,
        )

        simfile = parse_sm(output_path)
        print(f"Output: {output_path}")
        for chart in simfile.charts:
            stats = chart_pattern_stats(chart)
            print(
                f"  {chart.style} {chart.difficulty} ({chart.rating}) | "
                f"{int(stats['notes'])} notes | "
                f"jumps {int(stats['jumps'])} ({stats['jump_rate']:.2%}) | "
                f"jacks {int(stats['jacks'])} | "
                f"avg NPS {sum(n for _, n in chart.nps_series) / max(len(chart.nps_series), 1):.2f} | "
                f"peak NPS {stats['peak_nps']:.2f}"
            )


if __name__ == "__main__":
    main()
