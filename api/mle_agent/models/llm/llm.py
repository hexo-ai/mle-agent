from typing import AsyncGenerator

import openai

from ...config import get_settings
from ...helpers import log
from .types import ChatCompletionStreamed, ContentToken, OpenAIModels


class LLM:
    def __init__(self, openai_api_key: str | None = None):
        openai.api_key = get_settings().openai_api_key or openai_api_key

    async def ask_gpt(
        self,
        prompt: str,
        *,
        model: OpenAIModels = "gpt-3.5-turbo",
    ) -> AsyncGenerator[str, None]:
        output = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful coding assistant. \
You should always return back only code",
                },
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        assert isinstance(output, AsyncGenerator)
        async for response in output:
            validated_response = ChatCompletionStreamed.model_validate(response)
            if len(validated_response.choices) > 0 and isinstance(
                validated_response.choices[0].delta, ContentToken
            ):
                await log.ainfo(
                    "yielding",
                    content=validated_response.choices[0].delta.content,
                )
                yield validated_response.choices[0].delta.content
