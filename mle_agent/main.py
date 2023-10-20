from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mle_agent.models import LLM

models = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    print("Initializing LLM...")
    models["LLM"] = LLM()
    print("LLM initialized", models["LLM"])
    yield
    models.clear()


app = FastAPI(lifespan=lifespan)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponseContext(BaseModel):
    request: AskRequest


class AskResponse(BaseModel):
    answer: str
    context: AskResponseContext


@app.post("/chat/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if (llm := models.get("LLM")) is None:
        raise RuntimeError("LLM is not initialized")

    return AskResponse(
        answer=llm.ask_gpt(request.question),
        context=AskResponseContext(
            request=request,
        ),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
