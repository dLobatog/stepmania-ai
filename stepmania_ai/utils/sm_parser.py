"""Parser for StepMania .sm files.

Extracts metadata, timing info, and note data from .sm format simfiles.
Reference: https://github.com/stepmania/stepmania/wiki/sm
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# Note types in .sm format
NOTE_EMPTY = "0"
NOTE_TAP = "1"
NOTE_HOLD_HEAD = "2"
NOTE_HOLD_TAIL = "3"  # also roll tail
NOTE_ROLL_HEAD = "4"
NOTE_MINE = "M"

# Arrow indices: 0=left, 1=down, 2=up, 3=right
ARROW_NAMES = ["left", "down", "up", "right"]


@dataclass
class BPMChange:
    beat: float
    bpm: float


@dataclass
class Stop:
    beat: float
    duration: float  # seconds


@dataclass
class NoteRow:
    """A single row of notes at a specific time."""

    beat: float
    time: float  # in seconds, computed from BPM/offset
    arrows: str  # 4-char string like "1001"

    @property
    def is_empty(self) -> bool:
        return all(c == NOTE_EMPTY for c in self.arrows)

    @property
    def has_tap(self) -> bool:
        return NOTE_TAP in self.arrows or NOTE_HOLD_HEAD in self.arrows or NOTE_ROLL_HEAD in self.arrows

    @property
    def is_jump(self) -> bool:
        active = sum(1 for c in self.arrows if c in (NOTE_TAP, NOTE_HOLD_HEAD, NOTE_ROLL_HEAD))
        return active >= 2

    @property
    def tap_columns(self) -> list[int]:
        return [i for i, c in enumerate(self.arrows) if c in (NOTE_TAP, NOTE_HOLD_HEAD, NOTE_ROLL_HEAD)]


@dataclass
class Chart:
    """A single difficulty chart from a simfile."""

    style: str  # e.g. "dance-single"
    author: str
    difficulty: str  # e.g. "Challenge", "Expert", "Hard"
    rating: int
    note_rows: list[NoteRow] = field(default_factory=list)

    @property
    def taps_only(self) -> list[NoteRow]:
        return [r for r in self.note_rows if r.has_tap]

    @property
    def nps_series(self) -> list[tuple[float, float]]:
        """Notes per second over 1-second windows."""
        taps = self.taps_only
        if not taps:
            return []
        end_time = taps[-1].time
        result = []
        for t in range(int(end_time) + 1):
            count = sum(1 for r in taps if t <= r.time < t + 1)
            result.append((float(t), float(count)))
        return result


@dataclass
class Simfile:
    """Parsed .sm file."""

    path: Path
    title: str = ""
    artist: str = ""
    music_path: str = ""
    offset: float = 0.0
    sample_start: float = 0.0
    sample_length: float = 0.0
    bpms: list[BPMChange] = field(default_factory=list)
    stops: list[Stop] = field(default_factory=list)
    charts: list[Chart] = field(default_factory=list)

    @property
    def audio_path(self) -> Path:
        return self.path.parent / self.music_path

    def get_chart(self, difficulty: str = "Challenge", style: str = "dance-single") -> Chart | None:
        for c in self.charts:
            if c.difficulty.lower() == difficulty.lower() and c.style == style:
                return c
        return None


def _parse_tag(content: str, tag: str) -> str:
    """Extract a single #TAG:value; from .sm content."""
    pattern = rf"#{tag}:(.*?);"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _parse_bpms(raw: str) -> list[BPMChange]:
    """Parse BPM changes from 'beat=bpm,beat=bpm,...' format."""
    changes = []
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        beat_s, bpm_s = pair.split("=", 1)
        changes.append(BPMChange(beat=float(beat_s), bpm=float(bpm_s)))
    return sorted(changes, key=lambda c: c.beat)


def _parse_stops(raw: str) -> list[Stop]:
    """Parse stops from 'beat=duration,beat=duration,...' format."""
    stops = []
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        beat_s, dur_s = pair.split("=", 1)
        stops.append(Stop(beat=float(beat_s), duration=float(dur_s)))
    return sorted(stops, key=lambda s: s.beat)


def beat_to_time(beat: float, bpms: list[BPMChange], stops: list[Stop], offset: float) -> float:
    """Convert a beat number to a time in seconds, accounting for BPM changes and stops."""
    time = -offset
    prev_beat = 0.0
    prev_bpm = bpms[0].bpm if bpms else 120.0

    for change in bpms:
        if change.beat >= beat:
            break
        elapsed_beats = change.beat - prev_beat
        time += elapsed_beats * (60.0 / prev_bpm)
        prev_beat = change.beat
        prev_bpm = change.bpm

    # Add remaining beats at current BPM
    time += (beat - prev_beat) * (60.0 / prev_bpm)

    # Add stops that occur before this beat
    for stop in stops:
        if stop.beat <= beat:
            time += stop.duration

    return time


def _parse_chart(notes_block: str, bpms: list[BPMChange], stops: list[Stop], offset: float) -> Chart:
    """Parse a single #NOTES block into a Chart."""
    lines = notes_block.strip().split("\n")

    # First 5 lines are metadata (colon-separated)
    meta_lines = []
    note_lines = []
    in_notes = False
    for line in lines:
        line = line.strip()
        if not in_notes and len(meta_lines) < 5:
            meta_lines.append(line.rstrip(":").strip())
            if len(meta_lines) == 5:
                in_notes = True
        elif in_notes:
            note_lines.append(line)

    style = meta_lines[0] if len(meta_lines) > 0 else ""
    author = meta_lines[1] if len(meta_lines) > 1 else ""
    difficulty = meta_lines[2] if len(meta_lines) > 2 else ""
    rating = int(meta_lines[3]) if len(meta_lines) > 3 and meta_lines[3].isdigit() else 0

    # Parse measures separated by commas
    measures_raw = "\n".join(note_lines)
    # Remove comments
    measures_raw = re.sub(r"//.*", "", measures_raw)
    # Split by comma
    measures = [m.strip() for m in measures_raw.split(",")]

    note_rows = []
    for measure_idx, measure in enumerate(measures):
        rows = [r.strip() for r in measure.split("\n") if r.strip() and len(r.strip()) >= 4]
        if not rows:
            continue
        subdivisions = len(rows)
        for row_idx, row in enumerate(rows):
            arrow_data = row[:4]  # Only first 4 chars for dance-single
            beat = measure_idx * 4.0 + (row_idx / subdivisions) * 4.0
            time = beat_to_time(beat, bpms, stops, offset)
            note_rows.append(NoteRow(beat=beat, time=time, arrows=arrow_data))

    return Chart(
        style=style,
        author=author,
        difficulty=difficulty,
        rating=rating,
        note_rows=note_rows,
    )


def parse_sm(path: str | Path) -> Simfile:
    """Parse a .sm file and return a Simfile object."""
    path = Path(path)
    content = path.read_text(encoding="utf-8", errors="replace")

    title = _parse_tag(content, "TITLE")
    artist = _parse_tag(content, "ARTIST")
    music = _parse_tag(content, "MUSIC")
    offset = float(_parse_tag(content, "OFFSET") or "0")
    sample_start = float(_parse_tag(content, "SAMPLESTART") or "0")
    sample_length = float(_parse_tag(content, "SAMPLELENGTH") or "0")

    bpms = _parse_bpms(_parse_tag(content, "BPMS"))
    stops = _parse_stops(_parse_tag(content, "STOPS"))

    # Parse all #NOTES blocks
    charts = []
    notes_pattern = r"#NOTES:(.*?);"
    for match in re.finditer(notes_pattern, content, re.DOTALL):
        chart = _parse_chart(match.group(1), bpms, stops, offset)
        charts.append(chart)

    return Simfile(
        path=path,
        title=title,
        artist=artist,
        music_path=music,
        offset=offset,
        sample_start=sample_start,
        sample_length=sample_length,
        bpms=bpms,
        stops=stops,
        charts=charts,
    )


def main():
    """CLI: parse and summarize a .sm file."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: smai-parse <path_to.sm>")
        sys.exit(1)

    sm = parse_sm(sys.argv[1])
    print(f"Title: {sm.title}")
    print(f"Artist: {sm.artist}")
    print(f"Audio: {sm.audio_path}")
    print(f"Offset: {sm.offset}s")
    print(f"BPM changes: {len(sm.bpms)}")
    print(f"Stops: {len(sm.stops)}")
    print(f"Charts: {len(sm.charts)}")
    for chart in sm.charts:
        taps = chart.taps_only
        duration = taps[-1].time - taps[0].time if len(taps) > 1 else 0
        avg_nps = len(taps) / duration if duration > 0 else 0
        print(f"  {chart.style} {chart.difficulty} ({chart.rating}): "
              f"{len(taps)} notes, {avg_nps:.1f} avg NPS")


if __name__ == "__main__":
    main()
