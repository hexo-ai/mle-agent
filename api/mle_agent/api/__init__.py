from contextlib import asynccontextmanager
from typing import TypedDict, final

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..helpers import log
from ..models import LLM


@final
class Models(TypedDict, total=False):
    LLM: LLM


models: Models = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("Initializing LLM...")
    models["LLM"] = LLM()
    log.info("LLM initialized", llm_model=models["LLM"])
    yield
    models.clear()


class AskRequest(BaseModel):
    question: str


class AskResponseContext(BaseModel):
    request: AskRequest


class AskResponse(BaseModel):
    answer: str
    context: AskResponseContext


def init_api() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/chat/ask")
    async def ask(request: AskRequest) -> StreamingResponse:
        if (llm := models.get("LLM")) is None:
            raise RuntimeError("LLM is not initialized")

        return StreamingResponse(
            llm.ask_gpt(request.question),
            media_type="text/event-stream",
        )

    return app


if __name__ == "__main__":
    uvicorn.run(init_api(), host="0.0.0.0", port=8000)
