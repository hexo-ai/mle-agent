from typing import TypedDict

class CompletionUsage(TypedDict):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class ChatCompletionChoice(TypedDict):
    finish_reason: str
    index: int
    message: 'ChatCompletionMessage'

class ChatCompletion(TypedDict):
    id: str
    choices: list[ChatCompletionChoice]
    created: int
    model: str
    object: str
    usage: CompletionUsage | None

class ChatCompletionChunkChoiceDeltaFunctionCall(TypedDict):
    arguments: str | None
    name: str | None

class ChatCompletionChunkChoiceDelta(TypedDict):
    content: str | None
    function_call: ChatCompletionChunkChoiceDeltaFunctionCall | None
    role: str | None

class ChatCompletionChunkChoice(TypedDict):
    delta: ChatCompletionChunkChoiceDelta
    finish_reason: str | None
    index: int

class ChatCompletionChunk(TypedDict):
    id: str
    choices: list[ChatCompletionChunkChoice]
    created: int
    model: str
    object: str

class ChatCompletionMessageFunctionCall(TypedDict):
    arguments: str
    name: str

class ChatCompletionMessage(TypedDict):
    content: str | None
    role: str
    function_call: ChatCompletionMessageFunctionCall | None