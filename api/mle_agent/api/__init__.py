import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, TypedDict, final
from openai import OpenAI

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from pydantic import BaseModel
from langfuse import Langfuse
from langfuse.model import CreateTrace

from ..helpers import log
from ..models import LLM, InferencePipeline


def create_serialised_chunk(id: str, content: str) -> str:
    return (
        ChatCompletionChunk(
            id=id + uuid.uuid4().hex,
            choices=[
                Choice(
                    delta=ChoiceDelta(content=content),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=int(time.monotonic()),
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ).model_dump_json()
        + "\n"
    )


@final
class Models(TypedDict, total=False):
    LLM: LLM


models: Models = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    models.clear()


class AskRequest(BaseModel):
    repo_url: str
    branch: str | None = None
    messages: list[ChatCompletionMessageParam]
    model: str
    convId: str
    username: str
    userEmail: str


class AskResponseContext(BaseModel):
    request: AskRequest


class AskResponse(BaseModel):
    answer: str
    context: AskResponseContext


async def langfuse_trace(userId: str, username: str, userEmail: str):
    langfuse = Langfuse()
    trace = langfuse.trace(
        CreateTrace(
            # optional, if you want to use your own id
            id=userId,
            userId=userId,
            metadata={
                "env": "prod",
                "user": username,
                "email": userEmail,
            },
        )  # type: ignore
    )


async def get_response(request: AskRequest) -> AsyncGenerator[str, None]:
    pipeline = InferencePipeline(
        repo_url=request.repo_url,
        branch=request.branch,
    )

    async for content in pipeline.clone_and_process_repo():
        await asyncio.sleep(0.2)
        await langfuse_trace(request.convId, request.username, request.userEmail)

        serialised_chunk = create_serialised_chunk(
            id="repoProcess-",
            content=content,
        )
        await log.ainfo("messages", messages=request.messages)
        await log.ainfo("Sending chunk", chunk=serialised_chunk)
        yield serialised_chunk

    async for chunk in pipeline.get_response(
        messages=request.messages, model=request.model, user_id=request.convId
    ):
        yield chunk


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
        log.info(request.model)

        return StreamingResponse(
            get_response(request=request),
            media_type="text/event-stream",
        )

    return app


if __name__ == "__main__":
    uvicorn.run(init_api(), host="0.0.0.0", port=8000)
