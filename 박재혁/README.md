# PO3 Object Odyssey Prototype

This folder contains a demo-ready Python prototype for:

1. Building a character sheet from vision results and parent input
2. Generating active and soft style prompts
3. Creating two reference-image-based character variants
4. Writing a short children's story with OpenAI GPT
5. Showing everything in FastAPI + Streamlit

## Folder Layout

```text
박재혁/
  app/
  frontend/
  nukki/
  outputs/
  example_usage.py
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

`.env` example:

```env
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
FASTAPI_BASE_URL=http://127.0.0.1:8000
OPENAI_STORY_MODEL=gpt-4o-mini
GEMINI_TEXT_MODEL=gemini-2.0-flash
GEMINI_IMAGE_MODEL=gemini-2.5-flash-image
```

## Reference Images

Put JPG, JPEG, or PNG files into:

```text
박재혁/nukki
```

The backend will read the image list from that folder.

The default behavior uses the first available image.

## Run FastAPI

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Run Streamlit

Open a second terminal in the same folder and run:

```powershell
streamlit run frontend\streamlit_app.py
```

## Browser Demo

1. Start FastAPI
2. Start Streamlit
3. Open the Streamlit URL shown in the terminal
4. Enter the vision objects and parent input
5. Choose a reference image from `nukki`
6. Click `캐릭터 생성 시작`
7. Review the character sheet, prompts, generated images, and story

## Sample Input

```json
{
  "vision_result": {
    "objects": ["곰인형"]
  },
  "parent_input": {
    "name": "코코",
    "job": "우주 탐험가",
    "personality": "용감하고 다정함",
    "goal": "새로운 별을 찾고 싶어함",
    "extra_description": "작은 별 모양 가방을 메고 다녔으면 좋겠어"
  }
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

## Example Script

Run the full pipeline from the command line:

```powershell
python example_usage.py
```

Quick short demo:

```powershell
python quick_demo.py
```

## Notes

- The project does not implement the vision model itself.
- The Streamlit UI accepts demo vision input directly.
- The reference image path is validated to stay inside `박재혁/nukki`.
- Generated images are saved under `박재혁/outputs`.
