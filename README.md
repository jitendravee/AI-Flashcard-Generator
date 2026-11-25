# AI Flashcard Generator

A Streamlit application that converts student study notes into structured Question-Answer flashcards using the Google Gemini API, loads them into a `pandas` DataFrame, displays interactive revision cards, and exports them to CSV.

## Pipeline

```
Paste Notes → Gemini LLM returns Q&A pairs as JSON → Load into pandas DataFrame → Interactive Cards & CSV Download
```

## Features

- Paste study notes directly or upload `.txt` / `.md` files (up to 100 KB).
- Analyze note statistics (word count, sentence count, recommended flashcard count).
- Enforce strict JSON schema output from Gemini with grounding and prompt-injection guardrails.
- Multi-model fallback cascade with retry logic for transient `503` / `429` API errors.
- Load validated Q&A pairs into a `pandas.DataFrame` with `Card #`, `Question`, `Answer`, `Topic`, and `Difficulty` columns.
- Interactive flashcard grid with toggleable answer reveal, DataFrame table view, raw JSON inspector, and one-click CSV download (compatible with Anki, Quizlet, and Excel).

## Tech Stack

- Python 3
- Streamlit (`app.py`)
- Google Gen AI Python SDK (`google-genai`)
- pandas (`flashcard_utils.py`)
- python-dotenv
- unittest (`tests/`)

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your Gemini API key:

   ```env
   GEMINI_API_KEY=your_actual_api_key
   GEMINI_MODEL=gemini-2.5-flash-lite
   GEMINI_FALLBACK_MODELS=gemini-2.5-flash,gemini-2.0-flash,gemini-2.0-flash-lite
   ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

5. Run the unit test suite:

   ```bash
   python3 -m unittest discover -s tests -v
   ```

## Security

Never commit `.env` or share your Gemini API key publicly. `.gitignore` excludes `.env` and local virtual environments.
