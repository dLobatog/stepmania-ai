"""Isolated song-extraction helper used by the safe loader path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stepmania_ai.data.dataset import _cache_key, build_song_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a single song to cache")
    parser.add_argument("--sm-path", required=True)
    parser.add_argument("--difficulty", default="Challenge")
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()

    sm_path = Path(args.sm_path).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()
    data = build_song_data(sm_path, difficulty=args.difficulty, cache_dir=cache_dir)
    if data is None:
        return 0

    payload = {
        "n_frames": data["features"].n_frames,
        "n_onsets": int(data["onset_labels"].sum()),
        "title": data["title"],
        "path": str(sm_path),
        "cache_path": str(cache_dir / f"{_cache_key(sm_path)}_{sm_path.stem}.pkl"),
    }
    print(json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
