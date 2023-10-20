from typing import Literal

import openai
from pydantic import BaseModel

from mle_agent.config import get_settings

OpenAIModels = Literal["gpt-3.5-turbo", "gpt-4"]


class Message(BaseModel):
    content: str
    role: str


class Choice(BaseModel):
    finish_reason: str
    index: int
    message: Message


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatCompletion(BaseModel):
    choices: list[Choice]
    created: int
    id: str
    model: str
    object: str
    usage: Usage


class LLM:
    def __init__(self, openai_api_key: str | None = None):
        openai.api_key = get_settings().openai_api_key or openai_api_key

    def ask_gpt(
        self,
        prompt: str,
        *,
        model: OpenAIModels = "gpt-3.5-turbo",
    ) -> str:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful coding assistant. \
You should always return back only code",
                },
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        response_content = (
            ChatCompletion.model_validate(response).choices[0].message.content
        )
        return response_content
