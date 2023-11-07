from openai import AsyncOpenAI

from mle_agent.config import get_settings

openai_client = AsyncOpenAI(api_key=get_settings().openai_api_key)
