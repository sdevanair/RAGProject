"""Create image and text embeddings in the same pretrained CLIP vector space."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVENTS_FILE = PROJECT_ROOT / "data" / "events.json"
CLIP_MODEL_NAME = "clip-ViT-B-32"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def load_events(events_path=DEFAULT_EVENTS_FILE):
    """Read and validate the event list from JSON."""
    path = Path(events_path)
    with path.open(encoding="utf-8") as event_file:
        events = json.load(event_file)
    if not isinstance(events, list):
        raise ValueError(f"Expected a JSON list of events in {path}")
    return events


def resolve_thumbnail_path(thumbnail_path, project_root=None):
    """Resolve relative thumbnail paths from the project root."""
    root = Path(project_root) if project_root else PROJECT_ROOT
    raw = (thumbnail_path or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path


def is_valid_thumbnail(thumbnail_path, project_root=None):
    """Whether a thumbnail path points to an existing supported image."""
    resolved = resolve_thumbnail_path(thumbnail_path, project_root)
    return bool(
        resolved
        and resolved.is_file()
        and resolved.suffix.lower() in IMAGE_SUFFIXES
    )


def load_clip_model(model_name=CLIP_MODEL_NAME):
    """Load pretrained CLIP; the model downloads on first use if not cached."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_valid_event_images(events, model=None, project_root=None):
    """Encode existing thumbnails, returning aligned event IDs and vectors."""
    root = Path(project_root) if project_root else PROJECT_ROOT
    event_ids = []
    image_paths = []

    for event in events:
        path = resolve_thumbnail_path(event.get("thumbnail_path", ""), root)
        if (
            path is None
            or path.suffix.lower() not in IMAGE_SUFFIXES
            or not path.is_file()
        ):
            continue
        event_id = event.get("event_id", "")
        if not event_id:
            raise ValueError(f"Event with thumbnail {path} is missing event_id")
        event_ids.append(str(event_id))
        image_paths.append(path)

    if not image_paths:
        return [], np.empty((0, 0), dtype=np.float32)

    clip = model if model is not None else load_clip_model()
    images = []
    for path in image_paths:
        try:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not read event thumbnail {path}: {exc}") from exc

    embeddings = clip.encode(
        images,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(event_ids):
        raise RuntimeError(
            "CLIP returned an unexpected image embedding shape: "
            f"{embeddings.shape} for {len(event_ids)} images"
        )
    return event_ids, embeddings


def embed_texts(texts, model=None):
    """Encode natural-language text into CLIP's joint image/text space."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    clip = model if model is not None else load_clip_model()
    embeddings = clip.encode(
        list(texts),
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise RuntimeError(
            f"CLIP returned an unexpected text embedding shape: {embeddings.shape}"
        )
    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Generate CLIP image embeddings for event thumbnails."
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS_FILE)
    args = parser.parse_args()

    event_ids, embeddings = embed_valid_event_images(load_events(args.events))
    if not event_ids:
        raise SystemExit(
            "No valid event thumbnails found. Check thumbnail_path entries and image files."
        )
    print(f"CLIP model: {CLIP_MODEL_NAME}")
    print(f"Embedded {len(event_ids)} event(s): {', '.join(event_ids)}")
    print(f"Embedding shape: {embeddings.shape} (events x vector size)")


if __name__ == "__main__":
    main()
