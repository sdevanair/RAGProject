import json
import io
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.clip_embeddings import embed_texts, embed_valid_event_images
from src.evaluate_retrieval import EVALUATION_QUERIES, evaluate
from src.rag_answer import answer_query, build_context, generate_rag_answer, main as rag_main
from src.retrieve_events import retrieve_events
from src.clip_embeddings import PROJECT_ROOT


class FakeClipModel:
    def encode(self, values, convert_to_numpy=True, show_progress_bar=False):
        return np.tile(np.array([[1.0, 0.0, 0.5]], dtype=np.float32), (len(values), 1))


class FakeCollection:
    def __init__(self):
        self.query_arguments = None

    def count(self):
        return 1

    def query(self, **kwargs):
        self.query_arguments = kwargs
        return {
            "ids": [["evt_001"]],
            "distances": [[0.1]],
            "documents": [["Timestamp: 2026-10-14 | Object: car"]],
            "metadatas": [
                [
                    {
                        "event_id": "evt_001",
                        "object_class": "car",
                        "timestamp": "2026-10-14",
                    }
                ]
            ],
        }


class ClipEmbeddingTests(unittest.TestCase):
    def test_image_and_text_embeddings_have_aligned_rows_and_shared_dimension(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (16, 16), color="red").save(root / "event.jpg")
            events = [
                {
                    "event_id": "evt_001",
                    "thumbnail_path": "event.jpg",
                }
            ]
            model = FakeClipModel()

            event_ids, image_vectors = embed_valid_event_images(
                events, model=model, project_root=root
            )
            text_vectors = embed_texts(["a red car"], model=model)

            self.assertEqual(event_ids, ["evt_001"])
            self.assertEqual(image_vectors.shape, (1, 3))
            self.assertEqual(text_vectors.shape, (1, 3))

    def test_no_valid_thumbnails_does_not_load_model(self):
        event_ids, vectors = embed_valid_event_images(
            [{"event_id": "evt_001", "thumbnail_path": ""}],
            project_root=Path("."),
        )
        self.assertEqual(event_ids, [])
        self.assertEqual(vectors.shape, (0, 0))


class RetrievalTests(unittest.TestCase):
    def test_retrieval_uses_text_vector_and_returns_similarity(self):
        collection = FakeCollection()
        with patch(
            "src.retrieve_events.embed_texts",
            return_value=np.array([[1.0, 0.0, 0.5]], dtype=np.float32),
        ) as embed:
            results = retrieve_events(
                "cars by the road",
                top_k=3,
                collection=collection,
            )
        embed.assert_called_once()
        self.assertEqual(results[0]["event_id"], "evt_001")
        self.assertAlmostEqual(results[0]["similarity"], 0.9)
        self.assertEqual(collection.query_arguments["n_results"], 1)

    def test_blank_query_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            retrieve_events("  ", collection=FakeCollection())


class EvaluationTests(unittest.TestCase):
    def test_evaluation_reuses_one_model_and_reports_hit_rate(self):
        model = FakeClipModel()
        results = [
            {
                "event_id": "evt_001",
                "object_class": "car",
                "similarity": 0.3,
            }
        ]
        output = io.StringIO()
        with (
            patch("src.evaluate_retrieval.load_clip_model", return_value=model),
            patch("src.evaluate_retrieval.retrieve_events", return_value=results) as retrieve,
            redirect_stdout(output),
        ):
            metrics = evaluate(top_k=3)

        self.assertEqual(metrics["hit_rate"], 1.0)
        self.assertEqual(retrieve.call_count, len(EVALUATION_QUERIES))
        self.assertTrue(all(call.kwargs["model"] is model for call in retrieve.call_args_list))
        self.assertIn("Hit@3: 3/3", output.getvalue())


class ChromaIndexTests(unittest.TestCase):
    def test_real_chroma_collection_persists_event_and_retrieves_it(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            thumbnail_path = root / "evt_001.png"
            Image.new("RGB", (16, 16), color="blue").save(thumbnail_path)
            events_path = root / "events.json"
            events_path.write_text(
                json.dumps(
                    [
                        {
                            "event_id": "evt_001",
                            "timestamp": "2026-10-14T12:00:00",
                            "object_class": "car",
                            "description": "A car seen outdoors",
                            "thumbnail_path": str(thumbnail_path),
                            "confidence": 0.9,
                            "bounding_box": [1, 2, 3, 4],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            database_path = root / "chroma_db"
            script = """
import json
import numpy as np
import sys
from src.retrieve_events import retrieve_events
from src.vector_store import get_collection, index_events

class Model:
    def encode(self, values, convert_to_numpy=True, show_progress_bar=False):
        return np.tile(np.array([[1.0, 0.0, 0.5]], dtype=np.float32), (len(values), 1))

count = index_events(sys.argv[1], sys.argv[2], model=Model())
collection = get_collection(sys.argv[2])
result = retrieve_events("a blue vehicle", collection=collection, model=Model())
print(json.dumps({"count": count, "stored": collection.count(), "result": result[0]}))
"""
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(events_path),
                    str(database_path),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout.strip().splitlines()[-1])

            self.assertEqual(result["count"], 1)
            self.assertEqual(result["stored"], 1)
            self.assertEqual(result["result"]["event_id"], "evt_001")
            self.assertEqual(result["result"]["object_class"], "car")
            self.assertEqual(result["result"]["confidence"], 0.9)


class RagAnswerTests(unittest.TestCase):
    def test_context_contains_event_citation_and_evidence(self):
        context = build_context(
            [
                {
                    "event_id": "evt_001",
                    "timestamp": "2026-10-14T21:55:58",
                    "object_class": "car",
                    "description": "A car detected during motion",
                }
            ]
        )
        self.assertIn("[evt_001]", context)
        self.assertIn("A car detected during motion", context)

    def test_missing_api_key_does_not_return_a_fake_answer(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY is not set"):
                generate_rag_answer("What was seen?", [{"event_id": "evt_001"}])

    def test_missing_key_fails_before_retrieval(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("src.rag_answer.retrieve_events") as retrieve,
        ):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY is not set"):
                answer_query("What was seen?")
            retrieve.assert_not_called()

    def test_cli_requires_key_before_expensive_retrieval(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("src.rag_answer.retrieve_events") as retrieve,
            patch("sys.argv", ["rag_answer.py", "What was seen?"]),
        ):
            with self.assertRaises(SystemExit) as exc:
                rag_main()
            self.assertEqual(exc.exception.code, 2)
            retrieve.assert_not_called()

    def test_no_retrieved_events_needs_no_api_call(self):
        self.assertIn(
            "could not find any matching events",
            generate_rag_answer("What was seen?", []),
        )

    def test_api_answer_is_parsed_from_chat_completion(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "A car was detected [evt_001]."}}]}
                ).encode("utf-8")

        with patch("src.rag_answer.urlopen", return_value=FakeResponse()) as request:
            answer = generate_rag_answer(
                "What was seen?",
                [{"event_id": "evt_001", "object_class": "car"}],
                api_key="test-key",
            )
        self.assertEqual(answer, "A car was detected [evt_001].")
        request.assert_called_once()
        self.assertEqual(
            request.call_args.args[0].get_header("Authorization"),
            "Bearer test-key",
        )


    def test_ollama_answer_uses_local_openai_compatible_endpoint(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "A car was detected [evt_001]."}}]}
                ).encode("utf-8")

        with patch("src.rag_answer.urlopen", return_value=FakeResponse()) as request:
            answer = generate_rag_answer(
                "What was seen?",
                [{"event_id": "evt_001", "object_class": "car"}],
                provider="ollama",
            )

        self.assertEqual(answer, "A car was detected [evt_001].")
        sent_request = request.call_args.args[0]
        self.assertEqual(
            sent_request.full_url,
            "http://localhost:11434/v1/chat/completions",
        )
        self.assertEqual(sent_request.get_header("Authorization"), "Bearer ollama")
        self.assertEqual(json.loads(sent_request.data)["model"], "qwen3:4b")


if __name__ == "__main__":
    unittest.main()
