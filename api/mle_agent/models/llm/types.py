from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

OpenAIModels = Literal["gpt-3.5-turbo", "gpt-4"]


class Message(BaseModel):
    content: str
    role: str


class Choice(BaseModel):
    finish_reason: Optional[str]
    index: int


class NonStreamedChoice(Choice):
    message: Message


class RoleToken(BaseModel):
    role: str


class ContentToken(BaseModel):
    content: str


EmptyToken = BaseModel


class StreamedChoice(Choice):
    delta: Union[
        Message,
        RoleToken,
        ContentToken,
        EmptyToken,
    ] = Field(..., union_mode="left_to_right")


class ChatCompletion(BaseModel):
    choices: list[Union[NonStreamedChoice, StreamedChoice]]
    created: int
    id: str
    model: str
    object: str


class ChatCompletionStreamed(ChatCompletion):
    choices: list[StreamedChoice]


class ChatCompletionNonStreamed(ChatCompletion):
    choices: list[NonStreamedChoice]
