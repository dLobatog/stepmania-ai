#!/usr/bin/env python3
"""Generate evaluation charts and paper-friendly ablation summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from stepmania_ai.generate import generate_chart
from stepmania_ai.utils.sm_parser import parse_sm


def default_clips() -> list[Path]:
    clips = [
        Path("~/Downloads/migos_15s.mp3").expanduser(),
        Path("~/Downloads/migos_1min.mp3").expanduser(),
    ]
    return [clip for clip in clips if clip.exists()]


def pattern_entropy(patterns: list[str]) -> float:
    if not patterns:
        return 0.0
    counts = Counter(patterns)
    total = float(len(patterns))
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def chart_pattern_stats(chart) -> dict[str, float]:
    rows = chart.taps_only
    patterns = [row.arrows for row in rows]
    singles = sum(1 for row in rows if len(row.tap_columns) == 1)
    jumps = sum(1 for row in rows if len(row.tap_columns) >= 2)
    hands = sum(1 for row in rows if len(row.tap_columns) >= 3)
    jacks = 0
    repeated_patterns = 0
    staircase_runs = 0
    column_counts = [0, 0, 0, 0]

    for prev, curr in zip(rows, rows[1:]):
        if (
            len(prev.tap_columns) == 1
            and len(curr.tap_columns) == 1
            and prev.tap_columns[0] == curr.tap_columns[0]
            and (curr.time - prev.time) < 0.40
        ):
            jacks += 1
        if prev.arrows == curr.arrows and (curr.time - prev.time) < 0.50:
            repeated_patterns += 1

    for row in rows:
        for col in row.tap_columns:
            column_counts[col] += 1

    for a, b, c, d in zip(rows, rows[1:], rows[2:], rows[3:]):
        cols = [a.tap_columns, b.tap_columns, c.tap_columns, d.tap_columns]
        if all(len(step_cols) == 1 for step_cols in cols):
            seq = [step_cols[0] for step_cols in cols]
            if seq in ([0, 1, 2, 3], [3, 2, 1, 0]):
                staircase_runs += 1

    nps_values = [count for _, count in chart.nps_series]
    avg_nps = sum(nps_values) / max(len(nps_values), 1)
    max_column = max(column_counts) if column_counts else 0
    min_column = min(column_counts) if column_counts else 0
    return {
        "notes": float(len(rows)),
        "singles": float(singles),
        "jumps": float(jumps),
        "hands": float(hands),
        "jump_rate": jumps / max(len(rows), 1),
        "jacks": float(jacks),
        "repeat_rate": repeated_patterns / max(len(rows) - 1, 1),
        "staircase_runs": float(staircase_runs),
        "unique_patterns": float(len(set(patterns))),
        "pattern_entropy": pattern_entropy(patterns),
        "column_imbalance": (max_column - min_column) / max(len(rows), 1),
        "left_count": float(column_counts[0]),
        "down_count": float(column_counts[1]),
        "up_count": float(column_counts[2]),
        "right_count": float(column_counts[3]),
        "avg_nps": float(avg_nps),
        "peak_nps": float(max(nps_values) if nps_values else 0.0),
    }


def flatten_record(record: dict[str, object]) -> dict[str, object]:
    flat = {
        "clip": record["clip"],
        "strategy": record["strategy"],
        "output_path": str(record["output_path"]),
        "style": record["style"],
        "difficulty": record["difficulty"],
        "rating": record["rating"],
    }
    flat.update(record["stats"])
    return flat


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_strategy(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["strategy"])].append(row)

    summary: dict[str, dict[str, float]] = {}
    metric_names = [
        "notes",
        "singles",
        "jumps",
        "jump_rate",
        "jacks",
        "repeat_rate",
        "staircase_runs",
        "unique_patterns",
        "pattern_entropy",
        "column_imbalance",
        "avg_nps",
        "peak_nps",
    ]
    for strategy, group in grouped.items():
        summary[strategy] = {"clips": float(len(group))}
        for name in metric_names:
            values = [float(row[name]) for row in group]
            summary[strategy][name] = sum(values) / len(values) if values else 0.0
    return summary


def write_markdown_report(
    path: Path,
    rows: list[dict[str, object]],
    strategy_summary: dict[str, dict[str, float]],
) -> None:
    lines = [
        "# Evaluation Report",
        "",
        "This report compares decoder strategies on fixed audio clips so we can track playability-focused ablations over time.",
        "",
        "## Strategy Summary",
        "",
        "| strategy | clips | mean notes | mean jump rate | mean jacks | mean repeat rate | mean entropy | mean avg NPS | mean peak NPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for strategy, summary in sorted(strategy_summary.items()):
        lines.append(
            "| "
            f"{strategy} | "
            f"{summary['clips']:.0f} | "
            f"{summary['notes']:.1f} | "
            f"{summary['jump_rate']:.2%} | "
            f"{summary['jacks']:.2f} | "
            f"{summary['repeat_rate']:.2%} | "
            f"{summary['pattern_entropy']:.2f} | "
            f"{summary['avg_nps']:.2f} | "
            f"{summary['peak_nps']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Clip-Level Results",
            "",
            "| clip | strategy | notes | jumps | jacks | repeat rate | staircase runs | unique patterns | entropy | avg NPS | peak NPS | output |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for row in rows:
        lines.append(
            "| "
            f"{row['clip']} | "
            f"{row['strategy']} | "
            f"{float(row['notes']):.0f} | "
            f"{float(row['jumps']):.0f} | "
            f"{float(row['jacks']):.0f} | "
            f"{float(row['repeat_rate']):.2%} | "
            f"{float(row['staircase_runs']):.0f} | "
            f"{float(row['unique_patterns']):.0f} | "
            f"{float(row['pattern_entropy']):.2f} | "
            f"{float(row['avg_nps']):.2f} | "
            f"{float(row['peak_nps']):.2f} | "
            f"`{row['output_path']}` |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_clip(
    *,
    clip: Path,
    strategy: str,
    onset_model: str,
    pattern_model: str,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.name == strategy:
        clip_output = output_dir / f"{clip.stem}.sm"
    else:
        clip_output = output_dir / strategy / f"{clip.stem}.sm"
    clip_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Evaluating {clip.name} [{strategy}] ===")
    generate_chart(
        audio_path=clip,
        onset_model_path=onset_model,
        pattern_model_path=pattern_model,
        output_path=clip_output,
        title=clip.stem,
        decode_strategy=strategy,
        difficulty="Challenge",
        rating=10,
    )

    simfile = parse_sm(clip_output)
    chart = simfile.charts[0]
    stats = chart_pattern_stats(chart)
    print(
        f"Output: {clip_output}\n"
        f"  {chart.style} {chart.difficulty} ({chart.rating}) | "
        f"{int(stats['notes'])} notes | "
        f"jumps {int(stats['jumps'])} ({stats['jump_rate']:.2%}) | "
        f"jacks {int(stats['jacks'])} | "
        f"repeat rate {stats['repeat_rate']:.2%} | "
        f"entropy {stats['pattern_entropy']:.2f} | "
        f"avg NPS {stats['avg_nps']:.2f} | "
        f"peak NPS {stats['peak_nps']:.2f}"
    )
    return {
        "clip": clip.name,
        "strategy": strategy,
        "output_path": clip_output,
        "style": chart.style,
        "difficulty": chart.difficulty,
        "rating": chart.rating,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on fixed audio clips")
    parser.add_argument("--onset-model", required=True)
    parser.add_argument("--pattern-model", required=True)
    parser.add_argument("--output-dir", default="eval")
    parser.add_argument(
        "--decode-strategy",
        choices=["ergonomic", "raw"],
        help="Evaluate a single decoder. Omit to compare both raw and ergonomic decoding.",
    )
    parser.add_argument("clips", nargs="*", help="Audio clips to evaluate")
    args = parser.parse_args()

    clips = [Path(c).expanduser() for c in args.clips] if args.clips else default_clips()
    if not clips:
        raise SystemExit("No evaluation clips found. Pass clip paths explicitly.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = [args.decode_strategy] if args.decode_strategy else ["raw", "ergonomic"]

    records = []
    for strategy in strategies:
        for clip in clips:
            records.append(
                evaluate_clip(
                    clip=clip,
                    strategy=strategy,
                    onset_model=args.onset_model,
                    pattern_model=args.pattern_model,
                    output_dir=output_dir,
                )
            )

    flat_rows = [flatten_record(record) for record in records]
    strategy_summary = summarize_by_strategy(flat_rows)

    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    report_path = output_dir / "REPORT.md"

    json_path.write_text(
        json.dumps(
            {
                "strategies": strategies,
                "clips": [clip.name for clip in clips],
                "results": flat_rows,
                "strategy_summary": strategy_summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, flat_rows)
    write_markdown_report(report_path, flat_rows, strategy_summary)

    print(f"\nWrote JSON metrics to {json_path}")
    print(f"Wrote CSV metrics to {csv_path}")
    print(f"Wrote Markdown report to {report_path}")


if __name__ == "__main__":
    main()
