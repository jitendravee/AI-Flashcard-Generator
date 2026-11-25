import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from generator import GeminiFlashcardError, _extract_json, generate_demo_flashcards, generate_flashcards


class TestGenerator(unittest.TestCase):
    def test_extract_json_handles_markdown_fences(self):
        raw = '```json\n{"flashcards": [{"question": "Q?", "answer": "A."}]}\n```'
        parsed = _extract_json(raw)
        self.assertIn("flashcards", parsed)
        self.assertEqual(len(parsed["flashcards"]), 1)

    def test_extract_json_invalid_raises(self):
        with self.assertRaises(GeminiFlashcardError):
            _extract_json("not valid json at all")

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(GeminiFlashcardError):
                generate_flashcards("Some valid study notes about neural networks.", api_key=None)

    def test_fallback_on_503_unavailable(self):
        valid_payload = json.dumps(
            {
                "flashcards": [
                    {
                        "question": "Why divide by sqrt(d_k)?",
                        "answer": "To keep softmax gradients from vanishing.",
                        "topic": "Self-Attention",
                        "difficulty": "Medium",
                    }
                ]
            }
        )
        calls = []

        class FakeModels:
            def generate_content(self, model, contents, config):
                calls.append(model)
                if model == "primary-model":
                    raise RuntimeError("503 UNAVAILABLE: high demand")
                return SimpleNamespace(text=valid_payload)

        class FakeClient:
            def __init__(self, api_key):
                self.models = FakeModels()

            def close(self):
                pass

        with (
            patch.dict(
                "os.environ",
                {"GEMINI_MODEL": "primary-model", "GEMINI_FALLBACK_MODELS": "fallback-model"},
                clear=False,
            ),
            patch("generator.genai.Client", FakeClient),
            patch("generator.time.sleep", return_value=None),
        ):
            result = generate_flashcards(
                notes="Scaled dot-product attention divides by sqrt(d_k) to stabilize softmax.",
                num_cards=1,
                subject="Transformers",
                api_key="fake-key",
            )

        self.assertEqual(result["model_used"], "fallback-model")
        self.assertEqual(len(result["flashcards"]), 1)
        self.assertIn("primary-model", calls)
        self.assertIn("fallback-model", calls)

    def test_demo_flashcards_generation(self):
        notes = "1. Tokenization\nSplitting text into subword units for LLMs.\n\n2. Embeddings\nMapping tokens to dense vectors."
        result = generate_demo_flashcards(notes, num_cards=2, subject="NLP")
        self.assertEqual(len(result["flashcards"]), 2)
        self.assertEqual(result["model_used"], "offline-demo-parser")


if __name__ == "__main__":
    unittest.main()
