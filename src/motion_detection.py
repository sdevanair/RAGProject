"""Detect moving objects with MOG2 and YOLOv8n, logging events and thumbnails."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_ROOT / "data" / "input_video.mp4"
EVENTS_PATH = PROJECT_ROOT / "data" / "events.json"
THUMBNAILS_DIR = PROJECT_ROOT / "thumbnails"

MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 16
MOTION_MIN_AREA = 3000
WARMUP_FRAMES = 30

YOLO_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"
YOLO_CONFIDENCE = 0.45
YOLO_INFERENCE_INTERVAL_SECONDS = 1.0
COOLDOWN_SECONDS = 5.0
DUPLICATE_IOU_THRESHOLD = 0.5


def open_video(path):
    """Return a readable VideoCapture or raise an actionable error."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV could not open video: {path}")
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"OpenCV opened the video but could not read a frame: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap


def motion_detected(fg_mask, min_area):
    """True if the foreground mask contains a contour large enough to count."""
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(contour) >= min_area for contour in contours)


def box_iou(first, second):
    """Calculate intersection-over-union for [x1, y1, x2, y2] boxes."""
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def is_duplicate_detection(object_class, bounding_box, video_time, recent_events):
    """Match same-class boxes logged within the cooldown window."""
    return any(
        event["object_class"] == object_class
        and 0 <= video_time - event["_video_time"] < COOLDOWN_SECONDS
        and box_iou(bounding_box, event["bounding_box"]) >= DUPLICATE_IOU_THRESHOLD
        for event in recent_events
    )


def make_description(object_class, confidence):
    """Short text description for the event log."""
    article = "An" if object_class[0].lower() in "aeiou" else "A"
    return f"{article} {object_class} detected during motion (confidence {confidence:.0%})"


def load_yolo_model(weights_path=YOLO_WEIGHTS):
    """Load YOLOv8n weights locally and report model-loading failures clearly."""
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights_path}\n"
            "Place yolov8n.pt in the project folder, then run again."
        )
    try:
        return YOLO(str(weights_path))
    except Exception as exc:
        raise RuntimeError(f"Could not load YOLO model from {weights_path}: {exc}") from exc


def draw_detection(frame, bounding_box, object_class, confidence):
    """Draw one detection on a copy of its source frame."""
    annotated = frame.copy()
    x1, y1, x2, y2 = map(int, bounding_box)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        annotated,
        f"{object_class} {confidence:.2f}",
        (x1, max(y1 - 8, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return annotated


def save_events(events, path):
    """Write event records as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as event_file:
        json.dump(events, event_file, indent=2, ensure_ascii=False)
        event_file.write("\n")


def _thumbnail_reference(thumbnail_path):
    """Use project-relative paths when possible, otherwise retain an absolute path."""
    try:
        return thumbnail_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(thumbnail_path.resolve())


def process_video(
    video_path=VIDEO_PATH,
    events_path=EVENTS_PATH,
    thumbnails_dir=THUMBNAILS_DIR,
):
    """Process the video and return events containing real YOLO detections."""
    cap = open_video(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video: {video_path}")
        print(f"  Resolution: {width}x{height}, FPS: {fps:.2f}, frames: {frame_count}")

        thumbnails_dir = Path(thumbnails_dir)
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        back_sub = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=True,
        )
        print(f"Loading YOLO model: {YOLO_WEIGHTS}")
        model = load_yolo_model()

        events = []
        frame_index = 0
        event_counter = 0
        inference_interval_frames = max(
            1, round(fps * YOLO_INFERENCE_INTERVAL_SECONDS)
        )
        last_inference_frame = -inference_interval_frames
        base_time = datetime.now().replace(microsecond=0)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            fg_mask = back_sub.apply(frame)
            if frame_index >= WARMUP_FRAMES and motion_detected(
                fg_mask, MOTION_MIN_AREA
            ):
                if frame_index - last_inference_frame >= inference_interval_frames:
                    last_inference_frame = frame_index
                    video_time = frame_index / fps
                    results = model.predict(
                        frame,
                        conf=YOLO_CONFIDENCE,
                        verbose=False,
                    )
                    result = results[0]
                    if result.boxes is not None:
                        boxes = sorted(
                            result.boxes, key=lambda item: float(item.conf.item()), reverse=True
                        )
                        for box in boxes:
                            confidence = float(box.conf.item())
                            class_id = int(box.cls.item())
                            object_class = result.names[class_id]
                            bounding_box = [
                                round(float(coordinate), 2)
                                for coordinate in box.xyxy[0].tolist()
                            ]
                            if is_duplicate_detection(
                                object_class, bounding_box, video_time, events
                            ):
                                continue

                            event_counter += 1
                            event_id = f"evt_{event_counter:03d}"
                            thumbnail_path = thumbnails_dir / f"{event_id}.jpg"
                            annotated = draw_detection(
                                frame, bounding_box, object_class, confidence
                            )
                            if not cv2.imwrite(str(thumbnail_path), annotated):
                                raise OSError(f"Could not save thumbnail: {thumbnail_path}")

                            event = {
                                "event_id": event_id,
                                "timestamp": (
                                    base_time + timedelta(seconds=video_time)
                                ).isoformat(timespec="seconds"),
                                "object_class": object_class,
                                "description": make_description(
                                    object_class, confidence
                                ),
                                "thumbnail_path": _thumbnail_reference(thumbnail_path),
                                "bounding_box": bounding_box,
                                "confidence": round(confidence, 4),
                                "_video_time": video_time,
                            }
                            events.append(event)
                            print(
                                f"  Event {event_id}: {object_class} at "
                                f"{video_time:.1f}s -> {event['thumbnail_path']}"
                            )
            frame_index += 1

        saved_events = [
            {key: value for key, value in event.items() if key != "_video_time"}
            for event in events
        ]
        save_events(saved_events, events_path)
        print(
            f"\nProcessed {frame_index} frame(s); generated {len(saved_events)} "
            f"event(s) and saved {len(saved_events)} thumbnail(s)."
        )
        print(f"Events written to: {events_path}")
        return saved_events
    finally:
        cap.release()


def main():
    try:
        events = process_video()
    except (FileNotFoundError, RuntimeError, OSError, cv2.error) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    if not events:
        print(
            "No events were saved. Check that the video contains moving objects, "
            "or adjust MOTION_MIN_AREA, YOLO_CONFIDENCE, or "
            "YOLO_INFERENCE_INTERVAL_SECONDS."
        )


if __name__ == "__main__":
    main()
