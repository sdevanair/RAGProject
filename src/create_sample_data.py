"""
Create a small sample list of fictional video events for testing the RAG pipeline.

These are NOT real detections from a camera — they are made-up examples only.
"""

import json
from pathlib import Path


# Project folders: this file lives in src/, data/ is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "events.json"


def build_sample_events():
    """Return 5 fictional events with the fields required by the retrieval system."""
    return [
        {
            "event_id": "evt_001",
            "timestamp": "2026-03-15T09:12:00",
            "object_class": "person",
            "description": "A person walking near a parked car",
            "thumbnail_path": "",
        },
        {
            "event_id": "evt_002",
            "timestamp": "2026-03-15T14:30:45",
            "object_class": "person",
            "description": "A person entering a building",
            "thumbnail_path": "thumbnails/evt_002.jpg",
        },
        {
            "event_id": "evt_003",
            "timestamp": "2026-03-16T07:05:20",
            "object_class": "person",
            "description": "A person standing at a crosswalk waiting to cross",
            "thumbnail_path": "",
        },
        {
            "event_id": "evt_004",
            "timestamp": "2026-03-16T18:22:10",
            "object_class": "vehicle",
            "description": "A car stopping briefly in front of a gate",
            "thumbnail_path": "thumbnails/evt_004.jpg",
        },
        {
            "event_id": "evt_005",
            "timestamp": "2026-03-17T21:48:33",
            "object_class": "person",
            "description": "A person carrying a bag while walking along a sidewalk",
            "thumbnail_path": "",
        },
    ]


def save_events(events, output_path):
    """Write events to a JSON file with readable formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    events = build_sample_events()
    save_events(events, OUTPUT_FILE)
    print(f"Saved {len(events)} sample events to: {OUTPUT_FILE}")
    print("(Fictional data for testing - not from real video processing.)")


if __name__ == "__main__":
    main()
