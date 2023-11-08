"""
The Objective of this script is to build an infernece pipeline which can do the following:
7. Write the end to end logic involved as a function or class (NOT DONE)
8. Wrap this whole thing into a Flask application (NOT DONE)
9. Replace with OPENAI credits keys (NOT DONE)
10. Replace conda environment. Check if  the following code works (NOT DONE)
# Concatenate the dataframes and assign to corpus_df
corpus_df = pd.concat([corpus_df, chunks_with_embeddings_df, chunks_without_embeddings_df], ignore_index=True)
11. Fix long documents issue (HACKILY DONE)
12. Add import statements from the file (NOT DONE)
13. Integrate to a vector store (NOT DONE)
"""
import asyncio
import glob
import json
import textwrap
import time
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import aiofiles
import aiofiles.os
import numpy as np
import pandas as pd
import tiktoken
from git import GitCommandError, Repo
from openai.types.chat import ChatCompletionMessageParam

from ...helpers import log
from ...libs.openai import openai_client as openai
from .scripts.code_parser import TreeSitterPythonParser

FloatNDArray = np.ndarray[Any, np.dtype[np.float16]]


async def try_clone_repository(
    repo_url: str,
    *,
    repo_path: Path,
    branch: str,
):
    try:
        await log.ainfo(f"Cloning {( repo_url, branch ) = } to {repo_path = }")
        await asyncio.to_thread(
            Repo.clone_from,
            repo_url,
            repo_path,
            branch=branch,
            depth=1,
        )
        return True
    except GitCommandError as e:
        await log.awarning(f"Failed to clone {branch = }: {e}")
    return False


async def shallow_clone_repository(repo_url: str, repo_path: Path):
    if await aiofiles.os.path.exists(repo_path):
        return

    branches = ("master", "main")

    for branch in branches:
        if await try_clone_repository(repo_url, repo_path=repo_path, branch=branch):
            break
    else:
        raise ValueError(f"None of these {branches=} could be found in the repository.")


async def read_and_chunk_document(document_path: str, sem: asyncio.Semaphore):
    async with sem:
        async with aiofiles.open(document_path, "r") as file:
            try:
                document = await file.read()
            except UnicodeDecodeError:
                document = ""

    parser = TreeSitterPythonParser(document=document)
    [chunks, main_code, import_statements] = await asyncio.gather(
        asyncio.to_thread(parser.create_chunks),
        asyncio.to_thread(parser.extract_main_code),
        asyncio.to_thread(parser.extract_import_statements),
    )
    chunks_df = pd.DataFrame(chunks)
    new_rows = pd.DataFrame(
        [
            {"code": main_code, "type": "main_code"},
            {"code": import_statements, "type": "imports"},
        ]
    )
    chunks_df = pd.concat([chunks_df, new_rows], ignore_index=True)
    chunks_df = chunks_df[chunks_df["code"].apply(lambda x: len(x)) != 0]
    chunks_df["file_path"] = document_path
    return chunks_df


async def read_and_chunk_all_python_files(directory_path: Path):
    python_files = glob.glob(f"{directory_path}/**/*.py", recursive=True)
    sem = asyncio.Semaphore(100)
    chunks_df_ls = await asyncio.gather(
        *[read_and_chunk_document(file, sem) for file in python_files],
    )
    all_chunks_df: pd.DataFrame = pd.DataFrame()
    all_chunks_df = cast(pd.DataFrame, pd.concat(chunks_df_ls, ignore_index=True))
    return all_chunks_df


async def num_tokens(text: str, model: str, total_tokens: list[int]) -> int:
    """Return the number of tokens in a string."""
    encoding = await asyncio.to_thread(tiktoken.encoding_for_model, model)
    token_ls = await asyncio.to_thread(encoding.encode, text)
    num_tokens = len(token_ls)
    await log.ainfo("num_tokens", num_tokens=num_tokens, total_tokens=total_tokens[0])
    total_tokens[0] += 1000 if num_tokens > 8192 else num_tokens
    return num_tokens


def parse_embedding(embedding: str | list[str]):
    if isinstance(embedding, str):
        return json.loads(embedding)
    elif isinstance(embedding, list):
        return embedding
    else:
        raise ValueError(f"Unexpected value: {embedding}")


async def create_openai_embedding(
    batch: list[str],
    embedding_model_name: str,
    *,
    sem: asyncio.Semaphore,
    total_tokens: list[int],
    start_time: list[float],
) -> list[list[float]]:
    async with sem:
        batch = [
            item[:1000]
            if await num_tokens(
                item,
                embedding_model_name,
                total_tokens=total_tokens,
            )
            > 8192
            else item
            for item in batch
        ]

        await log.ainfo(
            "queue status",
            total_tokens=total_tokens[0],
        )

        # If the total number of tokens in a request exceeds 1M in <1m, sleep for 60 seconds
        current_time = time.time()
        if current_time - start_time[0] < 60 and total_tokens[0] >= 499_999:
            await log.ainfo("Sleeping for 60 seconds", total_tokens=total_tokens[0])
            await asyncio.sleep(60)
        elif current_time - start_time[0] > 60:
            start_time[0] = current_time + 60
            total_tokens[0] = 0

        response = await openai.embeddings.create(
            model=embedding_model_name, input=batch
        )
        await log.ainfo("response", response=len(response.data))
        for i, be in enumerate(response.data):
            assert i == be.index  # double check embeddings are in same order as input

    return [e.embedding for e in response.data]


async def create_openai_embeddings(
    inputs: list[str],
    embedding_model_name: str,
    batch_size=5,
) -> list[list[float]]:
    batches = [
        inputs[batch_start : batch_start + batch_size]
        for batch_start in range(0, len(inputs), batch_size)
    ]
    await log.ainfo(
        "batches", batches=[len(batch) for batch in batches], len=len(batches)
    )

    # the total number of tokens in a request must be less than 1M
    # to not hit the rate limit

    start = [time.time()]
    total_tokens = [0]
    sem = asyncio.Semaphore(50)
    embeddings_ls: list[list[list[float]]] = await asyncio.gather(
        *[
            create_openai_embedding(
                batch,
                embedding_model_name,
                total_tokens=total_tokens,
                start_time=start,
                sem=sem,
            )
            for batch in batches
        ]
    )

    return [embedding for embeddings in embeddings_ls for embedding in embeddings]


async def create_query_embedding(query: str, embedding_model_name: str):
    embeddings = await create_openai_embeddings([query], embedding_model_name)
    query_embedding: FloatNDArray = np.array(embeddings[0]).astype(float).reshape(1, -1)
    return query_embedding


def compute_cosine_similarity(
    chunks_embeddings: FloatNDArray, query_embedding: FloatNDArray
):
    log.info("embeddings", embeddings=chunks_embeddings)
    chunk_norms: FloatNDArray = np.linalg.norm(chunks_embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    # Compute cosine similarities
    similarities: FloatNDArray = np.dot(chunks_embeddings, query_embedding.T) / (
        chunk_norms[:, np.newaxis] * query_norm
    )
    return similarities


def create_message(query: str, context: str):
    message = f"Respond to the query based on the context provided: \
                If the query involves writing code, keep the code concise. \
                Write code only for what the user has asked for \
                Query: {query} \n Context: {context} \n Query: {query} \n Answer:"
    return message


def get_top_chunks(
    chunks: list[str],
    chunk_embeddings: FloatNDArray,
    query_embedding: FloatNDArray,
    top_n: int,
) -> tuple[list[str], list[FloatNDArray]]:
    similarities = compute_cosine_similarity(chunk_embeddings, query_embedding)
    top_indices = np.argsort(similarities.flatten())[-top_n:][
        ::-1
    ]  # Reverse the indices to get them in descending order
    top_chunks: list[str] = [chunks[i] for i in top_indices]
    top_similarities = [similarities.flatten()[i] for i in top_indices]
    return top_chunks, top_similarities


def add_imports_to_code(imports: list[str], code: str):
    code = textwrap.dedent(code)
    import_str = "\n".join(imports)
    import_str = textwrap.dedent(import_str)
    return import_str + "\n" + code


async def ask_gpt(query: str, context: str):
    message = create_message(query, context)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are an expert programmer"},
        {"role": "user", "content": message},
    ]
    model = "gpt-3.5-turbo"
    response = await openai.chat.completions.create(
        model=model, messages=messages, temperature=0, stream=True
    )
    async for chunk in response:
        yield chunk


class InferencePipeline:
    def __init__(
        self,
        repo_url: str,
        repo_parent_path: str | None = None,
        start_index_folder_path: str = "",
    ):
        self.repo_url = repo_url
        self.repo_parent_path = (
            Path(repo_parent_path) if repo_parent_path else Path.cwd() / "repos"
        )
        self.start_index_folder_path = Path(start_index_folder_path)
        columns = [
            "repo_url",
            "file_path",
            "code",
            "start_line_num",
            "end_line_num",
            "type",
            "parser_type",
            "embedding",
        ]
        self.corpus_df = pd.DataFrame(columns=columns)

    async def clone_and_process_repo(self):
        repo_path = Path.joinpath(
            self.repo_parent_path,
            *urlparse(self.repo_url).path.split(".")[0].split("/")[1:3],
        )
        repo_embedding_path = repo_path / "embeddings.csv"
        if await aiofiles.os.path.exists(repo_embedding_path):
            self.corpus_df = pd.read_csv(repo_embedding_path)
            return

        await log.ainfo("corpus_df", corpus_df=self.corpus_df)

        yield "Loading and processing repo...\n\n"

        await shallow_clone_repository(repo_url=self.repo_url, repo_path=repo_path)

        yield "Cloned repo...\n\n"

        await asyncio.sleep(0.5)

        yield "Processing repo...\n\n"

        all_chunks_df = await read_and_chunk_all_python_files(directory_path=repo_path)

        yield "Chunked repo...\n\n"

        await log.ainfo("all_chunks_df", all_chunks_df=all_chunks_df)

        chosen_types = ["class_definition", "function_definition"]

        chunks_with_embeddings_df = all_chunks_df[
            all_chunks_df["type"].isin(chosen_types)
        ]
        chunks_without_embeddings_df = all_chunks_df[
            ~all_chunks_df["type"].isin(chosen_types)
        ]

        yield "Creating embeddings...\n\n"

        chunks_with_embeddings_df.loc[:, "embedding"] = await create_openai_embeddings(
            chunks_with_embeddings_df["code"].tolist(),
            "text-embedding-ada-002",
            batch_size=10,
        )

        yield "Created embeddings...\n\n"

        # indicate end of repo processing
        yield ""

        self.corpus_df = pd.concat(
            [
                df
                for df in [
                    self.corpus_df,
                    chunks_with_embeddings_df,
                    chunks_without_embeddings_df,
                ]
                if not df.empty
            ],
            ignore_index=True,
        )
        self.corpus_df.to_csv(repo_embedding_path, index=False)

    async def compute_similarities(self, query: str):
        chosen_types = ["class_definition", "function_definition"]
        chunks_with_embeddings_df = cast(
            pd.DataFrame, self.corpus_df[self.corpus_df["type"].isin(chosen_types)]
        )
        chunks_with_import_statements = cast(
            pd.DataFrame,
            self.corpus_df[
                self.corpus_df["type"].isin(
                    ["import_statement", "import_from_statement"]
                )
            ],
        )

        chunk_embeddings = np.array(
            chunks_with_embeddings_df["embedding"].apply(parse_embedding).tolist()
        ).astype(float)
        query_embedding = await create_query_embedding(query, "text-embedding-ada-002")
        chunks = chunks_with_embeddings_df["code"].tolist()
        top_chunks, _ = get_top_chunks(
            chunks, chunk_embeddings, query_embedding, top_n=3
        )
        top_chunk_with_imports = add_imports_to_code(
            imports=chunks_with_import_statements["code"].tolist(), code=top_chunks[0]
        )
        return top_chunks, top_chunk_with_imports

    async def get_response(self, query: str):
        top_chunks, _ = await self.compute_similarities(query=query)
        async for sample_response in ask_gpt(query, context="\n".join(top_chunks[:2])):
            await asyncio.sleep(0.01)
            yield sample_response.model_dump_json() + "\n"


if __name__ == "__main__":
    pipeline = InferencePipeline(
        repo_url="https://github.com/langchain-ai/langchain.git",
        repo_parent_path="samples",
        start_index_folder_path="langchain/libs/langchain/langchain/document_transformers",
    )
    pipeline.clone_and_process_repo()
    query = "Write python function to read a HTML file and transform it into text using Langchain"
    pipeline.get_response(query=query)
