"""Streamlit web application for converting study notes into Q&A flashcards and CSV."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from flashcard_utils import (
    FlashcardDataError,
    analyze_notes,
    dataframe_to_csv_bytes,
    flashcards_to_dataframe,
)
from generator import GeminiFlashcardError, generate_demo_flashcards, generate_flashcards

load_dotenv()

st.set_page_config(
    page_title="AI Flashcard Generator",
    page_icon="🗂️",
    layout="wide",
)

st.title("🗂️ AI Flashcard Generator")
st.write(
    "Convert study notes into concise Question-Answer flashcards for quick revision. "
    "Paste your notes below, let Gemini return structured Q&A pairs as JSON, inspect them "
    "as interactive cards and a pandas DataFrame, and export directly to CSV."
)

SAMPLE_NOTES_PATH = Path(__file__).parent / "sample_notes.txt"

with st.sidebar:
    st.header("Flashcard settings")
    subject = st.text_input("Subject / Topic", value="Generative AI & LLMs", placeholder="e.g. Operating Systems")
    num_cards = st.slider("Number of flashcards", min_value=3, max_value=15, value=6, step=1)
    difficulty = st.selectbox("Difficulty level", options=["Mixed", "Easy", "Medium", "Hard"], index=0)
    demo_mode = st.toggle(
        "Offline Demo Mode",
        value=False,
        help="Run a local deterministic parser when no GEMINI_API_KEY is configured.",
    )
    st.divider()
    if st.button("Load sample notes", use_container_width=True):
        if SAMPLE_NOTES_PATH.exists():
            st.session_state["notes_input"] = SAMPLE_NOTES_PATH.read_text(encoding="utf-8")
    st.caption("The Gemini API key is read from your local .env file. Do not commit .env to GitHub.")

uploaded_file = st.file_uploader("Upload notes file (optional .txt / .md)", type=["txt", "md"])
if uploaded_file is not None:
    try:
        st.session_state["notes_input"] = uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.error("Uploaded file must be UTF-8 encoded.")

notes_text = st.text_area(
    "Paste your study notes",
    key="notes_input",
    height=220,
    placeholder="Paste lecture notes, textbook summaries, or key definitions here...",
)

if notes_text.strip():
    try:
        stats = analyze_notes(notes_text)
        col1, col2, col3 = st.columns(3)
        col1.metric("Word count", stats["word_count"])
        col2.metric("Sentences / lines", stats["sentence_count"])
        col3.metric("Recommended cards", stats["recommended_cards"])
    except FlashcardDataError as exc:
        st.warning(str(exc))

if st.button("Generate Flashcards", type="primary", use_container_width=True):
    try:
        stats = analyze_notes(notes_text)
        with st.spinner("Generating structured Q&A flashcards..."):
            if demo_mode:
                result = generate_demo_flashcards(
                    notes=stats["cleaned_text"],
                    num_cards=num_cards,
                    subject=subject.strip() or "General",
                )
            else:
                result = generate_flashcards(
                    notes=stats["cleaned_text"],
                    num_cards=num_cards,
                    subject=subject.strip() or "General",
                    difficulty=difficulty,
                )

            df = flashcards_to_dataframe(result["flashcards"])
            csv_bytes = dataframe_to_csv_bytes(df)

        st.session_state["generated_flashcards"] = {
            "cards": result["flashcards"],
            "df": df,
            "csv_bytes": csv_bytes,
            "raw_json": result["raw_json"],
            "model_used": result["model_used"],
            "subject": subject.strip() or "flashcards",
        }
    except (FlashcardDataError, GeminiFlashcardError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")

if "generated_flashcards" in st.session_state:
    data = st.session_state["generated_flashcards"]
    df = data["df"]
    cards = data["cards"]

    st.divider()
    st.subheader("Generated Revision Flashcards")
    st.success(
        f"Generated {len(cards)} Q&A flashcard(s) loaded into a pandas DataFrame "
        f"(model: `{data['model_used']}`)."
    )

    cards_tab, table_tab, json_tab = st.tabs(["Flashcards View", "DataFrame & CSV Export", "Raw JSON Output"])

    with cards_tab:
        show_all = st.checkbox("Reveal all answers", value=True)
        cols = st.columns(2)
        for idx, card in enumerate(cards, start=1):
            with cols[(idx - 1) % 2]:
                with st.container(border=True):
                    st.caption(f"Card {idx} • {card['topic']} • {card['difficulty']}")
                    st.markdown(f"**Q:** {card['question']}")
                    with st.expander("Show Answer", expanded=show_all):
                        st.markdown(f"**A:** {card['answer']}")

    with table_tab:
        st.dataframe(df, use_container_width=True, hide_index=True)
        safe_slug = "".join(c if c.isalnum() else "_" for c in data["subject"].lower()).strip("_") or "flashcards"
        st.download_button(
            label="⬇️ Download Flashcards as CSV",
            data=data["csv_bytes"],
            file_name=f"{safe_slug}_flashcards.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )
        with st.expander("Preview CSV text"):
            st.code(data["csv_bytes"].decode("utf-8"), language="csv")

    with json_tab:
        st.code(json.dumps(data["raw_json"], indent=2, ensure_ascii=False), language="json")
