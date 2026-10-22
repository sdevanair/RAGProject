"""Small, transparent retrieval check against the project's real event labels."""

import argparse
from pathlib import Path

from src.clip_embeddings import DEFAULT_EVENTS_FILE, load_clip_model, load_events
from src.retrieve_events import retrieve_events
from src.vector_store import DEFAULT_DB_PATH

EVALUATION_QUERIES = (
    "Show me cars detected in the video.",
    "Find a photograph of a car or other vehicle.",
    "Which event images contain an automobile?",
)


def evaluate(top_k=5, persist_directory=DEFAULT_DB_PATH, events_path=DEFAULT_EVENTS_FILE):
    """Report hit@k on representative queries using labeled event classes."""
    events = load_events(events_path)
    expected_classes = {
        str(event["event_id"]): str(event.get("object_class", "")).casefold()
        for event in events
        if event.get("event_id")
    }
    if not expected_classes:
        raise ValueError(f"No labeled event IDs found in {events_path}")

    model = load_clip_model()
    hit_count = 0
    scores = []
    for query in EVALUATION_QUERIES:
        results = retrieve_events(query, top_k, persist_directory, model=model)
        hits = [
            result for result in results
            if expected_classes.get(str(result["event_id"])) == "car"
        ]
        hit = bool(hits)
        hit_count += int(hit)
        if results:
            scores.append(float(results[0]["similarity"]))
        print(f"\nQuery: {query}")
        print(f"  Hit@{top_k} for class 'car': {hit}")
        for result in results:
            print(
                f"  {result['event_id']}: {result.get('object_class')} "
                f"(CLIP cosine similarity {result['similarity']:.4f})"
            )

    hit_rate = hit_count / len(EVALUATION_QUERIES)
    mean_top_similarity = sum(scores) / len(scores) if scores else 0.0
    print(
        f"\nHit@{top_k}: {hit_count}/{len(EVALUATION_QUERIES)} "
        f"({hit_rate:.1%}); mean top-result similarity: {mean_top_similarity:.4f}"
    )
    print(
        "Evaluation labels come from object_class in events.json. The current "
        "video has only car labels, so this is a small smoke evaluation, not a "
        "balanced benchmark."
    )
    return {"hit_rate": hit_rate, "mean_top_similarity": mean_top_similarity}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate semantic retrieval against current event labels."
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS_FILE)
    args = parser.parse_args()
    evaluate(args.top_k, args.db, args.events)


if __name__ == "__main__":
    main()
