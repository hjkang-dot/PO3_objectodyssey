# PO3 Object Odyssey Prototype

This folder contains a demo-ready Python prototype for:

1. Building a character sheet from vision results and parent input
2. Generating active and soft style prompts
3. Creating two reference-image-based character variants
4. Generating a short story package with title, story paragraphs, TTS script, and choices
5. Showing everything in FastAPI + Streamlit

## Folder Layout

```text
박재혁/
  app/
  frontend/
  nukki/
  outputs/
  example_usage.py
  example_story_usage.py
  requirements.txt
  README.md
  .env.example
```

## Move Into The Project

```powershell
cd C:\Users\user\Desktop\interactiveLLM\박재혁
```

## Create And Activate A Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## Install Dependencies

```powershell
pip install -r requirements.txt
```

## Environment Variables

1. Copy `.env.example` to `.env`
2. Fill in your API keys

```powershell
copy .env.example .env
```

Example:

```env
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
FASTAPI_BASE_URL=http://127.0.0.1:8000
OPENAI_STORY_MODEL=gpt-4o-mini
GEMINI_TEXT_MODEL=gemini-2.5-flash
GEMINI_IMAGE_MODEL=gemini-2.5-flash-image
```

## Reference Images

Put JPG, JPEG, or PNG files into:

```text
박재혁/nukki
```

## Run FastAPI

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Run Streamlit

```powershell
streamlit run frontend\streamlit_app.py
```

## Story Generation Only

Generate only the story package from a sample character sheet:

```powershell
python example_story_usage.py
```

The returned JSON follows this structure:

```json
{
  "title": "string",
  "story_paragraphs": ["string", "string", "string"],
  "tts_script": [{"line": "string", "tone": "string"}],
  "choices": [
    {"id": "string", "text": "string"},
    {"id": "string", "text": "string"}
  ]
}
```

## API Endpoints

- `GET /health`
- `GET /reference-images`
- `POST /character-sheet`
- `POST /style-prompts`
- `POST /generate-images`
- `POST /generate-story`
- `POST /pipeline`

## Notes

- Story generation uses `character_sheet` as input and does not create a new protagonist.
- Story tone must be one of `따뜻한`, `모험적인`, or `교훈적인`.
- TTS audio synthesis is not implemented here, but the `tts_script` structure is ready for future integration.
