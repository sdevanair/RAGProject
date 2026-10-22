"""Persistent ChromaDB index for CLIP-embedded event thumbnails."""

import argparse
from pathlib import Path

from src.clip_embeddings import (
    CLIP_MODEL_NAME,
    DEFAULT_EVENTS_FILE,
    PROJECT_ROOT,
    embed_valid_event_images,
    load_events,
)

DEFAULT_DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "video_events"


def _load_chromadb():
    """Import Chroma only when the persistent database is used."""
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "ChromaDB is not installed. Install it with "
            "'.\\.venv\\Scripts\\python.exe -m pip install chromadb'."
        ) from exc
    return chromadb


def get_collection(persist_directory=DEFAULT_DB_PATH):
    """Open or create the persistent collection using cosine distance."""
    chromadb = _load_chromadb()
    client = chromadb.PersistentClient(path=str(Path(persist_directory)))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": CLIP_MODEL_NAME},
    )


def _event_document(event):
    """Build a readable text record for results and later RAG context."""
    return " | ".join(
        part
        for part in (
            f"Timestamp: {event.get('timestamp', 'unknown')}",
            f"Object: {event.get('object_class', 'unknown')}",
            event.get("description", ""),
        )
        if part
    )


def _event_metadata(event):
    """Convert event fields to Chroma-supported scalar metadata."""
    metadata = {
        "event_id": str(event["event_id"]),
        "timestamp": str(event.get("timestamp", "")),
        "object_class": str(event.get("object_class", "unknown")),
        "description": str(event.get("description", "")),
        "thumbnail_path": str(event.get("thumbnail_path", "")),
    }
    if "confidence" in event:
        metadata["confidence"] = float(event["confidence"])
    if "bounding_box" in event:
        metadata["bounding_box"] = ",".join(
            str(float(value)) for value in event["bounding_box"]
        )
    return metadata


def index_events(
    events_path=DEFAULT_EVENTS_FILE,
    persist_directory=DEFAULT_DB_PATH,
    model=None,
):
    """Embed real thumbnails and synchronize event metadata into ChromaDB."""
    events = load_events(events_path)
    events_by_id = {
        str(event["event_id"]): event
        for event in events
        if event.get("event_id")
    }
    event_ids, vectors = embed_valid_event_images(events, model=model)
    if not event_ids:
        raise ValueError(
            "No events with readable thumbnail paths were found; "
            "the existing Chroma collection was left unchanged."
        )

    collection = get_collection(persist_directory)
    collection.upsert(
        ids=event_ids,
        embeddings=vectors.tolist(),
        documents=[_event_document(events_by_id[event_id]) for event_id in event_ids],
        metadatas=[_event_metadata(events_by_id[event_id]) for event_id in event_ids],
    )

    stale_ids = set(collection.get()["ids"]) - set(event_ids)
    if stale_ids:
        collection.delete(ids=sorted(stale_ids))
    return collection.count()


def main():
    parser = argparse.ArgumentParser(
        description="Build or refresh the persistent CLIP event index."
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS_FILE)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    count = index_events(args.events, args.db)
    print(f"Indexed {count} event(s) in {args.db}")
    print(f"Embedding model: {CLIP_MODEL_NAME}")


if __name__ == "__main__":
    main()