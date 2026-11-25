"""Utilities for note analysis, flashcard JSON validation, DataFrame creation, and CSV export."""

from __future__ import annotations

import io
import re
from typing import Any

import pandas as pd

MAX_NOTES_BYTES = 100_000
MIN_NOTES_CHARS = 30
REQUIRED_COLUMNS = ["Card #", "Question", "Answer", "Topic", "Difficulty"]


class FlashcardDataError(Exception):
    """Raised when study notes or generated flashcard payloads are invalid."""


def analyze_notes(notes_text: str) -> dict[str, Any]:
    """Validate pasted study notes and return basic text statistics.

    Args:
        notes_text: Raw study notes pasted or uploaded by the student.

    Returns:
        A dictionary containing cleaned notes, word count, sentence count,
        character count, and a recommended number of flashcards.

    Raises:
        FlashcardDataError: If the notes are empty, too short, or exceed size limits.
    """
    if not isinstance(notes_text, str):
        raise FlashcardDataError("Notes must be provided as text.")

    cleaned = notes_text.strip()
    if len(cleaned) < MIN_NOTES_CHARS:
        raise FlashcardDataError(
            f"Please paste at least {MIN_NOTES_CHARS} characters of study notes to generate flashcards."
        )

    if len(cleaned.encode("utf-8")) > MAX_NOTES_BYTES:
        raise FlashcardDataError("Notes exceed the 100 KB limit. Split them into smaller sections and try again.")

    words = re.findall(r"\b\w+\b", cleaned)
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", cleaned) if s.strip()]
    recommended = max(3, min(12, len(words) // 40 + 3))

    return {
        "cleaned_text": cleaned,
        "char_count": len(cleaned),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "recommended_cards": recommended,
    }


def normalize_flashcards(
    raw_cards: list[dict[str, Any]],
    default_topic: str = "General",
    default_difficulty: str = "Medium",
) -> list[dict[str, str]]:
    """Validate, clean, and deduplicate a list of Q&A flashcard dictionaries.

    Args:
        raw_cards: List of raw flashcard objects parsed from LLM JSON output.
        default_topic: Fallback topic label if a card omits the topic field.
        default_difficulty: Fallback difficulty label if omitted or invalid.

    Returns:
        A list of normalized flashcard dictionaries with consistent keys.

    Raises:
        FlashcardDataError: If no valid Q&A pairs remain after validation.
    """
    if not isinstance(raw_cards, list) or not raw_cards:
        raise FlashcardDataError("No flashcards were returned. Please try again with more detailed notes.")

    allowed_difficulties = {"Easy", "Medium", "Hard"}
    normalized: list[dict[str, str]] = []
    seen_questions: set[str] = set()

    for item in raw_cards:
        if not isinstance(item, dict):
            continue

        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue

        dedup_key = re.sub(r"\s+", " ", question.lower())
        if dedup_key in seen_questions:
            continue
        seen_questions.add(dedup_key)

        topic = str(item.get("topic", "")).strip() or default_topic
        raw_diff = str(item.get("difficulty", "")).strip().capitalize()
        difficulty = raw_diff if raw_diff in allowed_difficulties else default_difficulty

        normalized.append(
            {
                "question": question,
                "answer": answer,
                "topic": topic,
                "difficulty": difficulty,
            }
        )

    if not normalized:
        raise FlashcardDataError("The generated response contained no valid question-answer pairs.")

    return normalized


def flashcards_to_dataframe(flashcards: list[dict[str, str]]) -> pd.DataFrame:
    """Convert normalized flashcard dictionaries into a structured pandas DataFrame.

    Args:
        flashcards: List of normalized flashcard dicts.

    Returns:
        A pandas DataFrame with columns: Card #, Question, Answer, Topic, Difficulty.
    """
    rows = [
        {
            "Card #": idx,
            "Question": card["question"],
            "Answer": card["answer"],
            "Topic": card.get("topic", "General"),
            "Difficulty": card.get("difficulty", "Medium"),
        }
        for idx, card in enumerate(flashcards, start=1)
    ]
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a flashcard DataFrame to UTF-8 CSV bytes for download.

    Args:
        df: Flashcard pandas DataFrame.

    Returns:
        UTF-8 encoded CSV bytes (without index column).
    """
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")
