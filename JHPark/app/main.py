"""FastAPI application entrypoint."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.utils import PROJECT_ROOT

APP_ROOT = PROJECT_ROOT / "JHPark"

load_dotenv(APP_ROOT / ".env")

app = FastAPI(title="PO3 Object Odyssey Prototype", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    """Provide a minimal root endpoint."""

    return {"message": "PO3 Object Odyssey prototype API is running."}


def run() -> None:
    """Run the FastAPI app with Uvicorn."""

    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    run()
