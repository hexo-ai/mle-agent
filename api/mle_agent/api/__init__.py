from contextlib import asynccontextmanager
from typing import TypedDict, final

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..helpers import log
from ..models import LLM, InferencePipeline


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
    repo_url: str
    query: str
    start_index_folder_path: str = ""
    id: str | None = None
    response_id: str | None = None


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

    @app.get("/")
    async def health_check():
        return {"it": "works"}

    @app.post("/chat/ask")
    async def ask(request: AskRequest) -> StreamingResponse:
        log.info(request.repo_url)
        log.info(request.start_index_folder_path)
        pipeline = InferencePipeline(
            repo_url=request.repo_url,
            start_index_folder_path=request.start_index_folder_path,
        )
        await pipeline.clone_and_process_repo()
        return StreamingResponse(
            pipeline.get_response(query=request.query),
            media_type="text/event-stream",
        )

    return app


if __name__ == "__main__":
    uvicorn.run(init_api(), host="0.0.0.0", port=8000)
