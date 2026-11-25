import unittest

import pandas as pd

from flashcard_utils import (
    FlashcardDataError,
    analyze_notes,
    dataframe_to_csv_bytes,
    flashcards_to_dataframe,
    normalize_flashcards,
)


class TestFlashcardUtils(unittest.TestCase):
    def test_analyze_notes_valid(self):
        notes = (
            "Self-attention allows every token in a sequence to attend to every other token "
            "in parallel using Query, Key, and Value matrices."
        )
        stats = analyze_notes(notes)
        self.assertGreater(stats["word_count"], 15)
        self.assertGreaterEqual(stats["recommended_cards"], 3)

    def test_analyze_notes_too_short(self):
        with self.assertRaises(FlashcardDataError):
            analyze_notes("Too short")

    def test_normalize_flashcards_deduplicates_and_cleans(self):
        raw = [
            {"question": "What is Self-Attention?", "answer": "Mechanism linking tokens.", "topic": "Transformers", "difficulty": "easy"},
            {"question": "what is   self-attention?", "answer": "Duplicate entry.", "topic": "Transformers", "difficulty": "Hard"},
            {"question": "", "answer": "Missing question"},
            {"question": "What is Temperature?", "answer": "Controls sampling randomness.", "difficulty": "unknown"},
        ]
        cleaned = normalize_flashcards(raw, default_topic="LLMs", default_difficulty="Medium")
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0]["difficulty"], "Easy")
        self.assertEqual(cleaned[1]["topic"], "LLMs")
        self.assertEqual(cleaned[1]["difficulty"], "Medium")

    def test_dataframe_and_csv_export(self):
        cards = [
            {"question": "Q1?", "answer": "A1, with comma.", "topic": "Basics", "difficulty": "Easy"},
            {"question": "Q2?", "answer": "A2.", "topic": "Advanced", "difficulty": "Hard"},
        ]
        df = flashcards_to_dataframe(cards)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["Card #", "Question", "Answer", "Topic", "Difficulty"])
        self.assertEqual(len(df), 2)

        csv_text = dataframe_to_csv_bytes(df).decode("utf-8")
        self.assertIn("Card #,Question,Answer,Topic,Difficulty", csv_text)
        self.assertIn('"A1, with comma."', csv_text)


if __name__ == "__main__":
    unittest.main()
