"""Find event thumbnails that are semantically similar to a text query."""

import argparse
import json
from pathlib import Path

from src.clip_embeddings import DEFAULT_EVENTS_FILE, PROJECT_ROOT, embed_texts
from src.vector_store import DEFAULT_DB_PATH, get_collection


def retrieve_events(
    query,
    top_k=5,
    persist_directory=DEFAULT_DB_PATH,
    model=None,
    collection=None,
):
    """Search ChromaDB with a CLIP text embedding; results are never keyword-based."""
    query = query.strip()
    if not query:
        raise ValueError("The search query must not be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    store = collection if collection is not None else get_collection(persist_directory)
    count = store.count()
    if count == 0:
        return []

    query_vector = embed_texts([query], model=model)[0].tolist()
    matches = store.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    result_ids = matches.get("ids", [[]])[0]
    distances = matches.get("distances", [[]])[0]
    documents = matches.get("documents", [[]])[0]
    metadatas = matches.get("metadatas", [[]])[0]

    results = []
    for event_id, distance, document, metadata in zip(
        result_ids, distances, documents, metadatas
    ):
        results.append(
            {
                **(metadata or {}),
                "event_id": (metadata or {}).get("event_id", event_id),
                "document": document or "",
                "distance": float(distance),
                "similarity": 1.0 - float(distance),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve video events using CLIP semantic similarity."
    )
    parser.add_argument("query", help="Natural-language description of the event")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    results = retrieve_events(args.query, args.top_k, args.db)
    if not results:
        print("The event index is empty. Build it first with: python -m src.vector_store")
        return
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()