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
import os
import glob
import json
from ...config import get_settings
import openai
import tiktoken
import textwrap
import numpy as np
import pandas as pd

from typing import Any, TypedDict, cast
from pathlib import Path
from dotenv import load_dotenv
from git import Repo, GitCommandError
from .scripts.code_parser import TreeSitterPythonParser
from .types import ChatCompletionChunk

FloatNDArray = np.ndarray[Any, np.dtype[np.float16]]

openai.api_key = get_settings().openai_api_key


def try_clone_repo(
    repo_url: str,
    repo_path: str,
    *,
    branches=("master", "main"),
    depth=1,
):
    for branch in branches:
        try:
            Repo.clone_from(repo_url, repo_path, branch=branch, depth=depth)
            return  # Exit the function once a successful clone operation is performed
        except GitCommandError as e:
            print(f"Failed to clone branch {branch}: {e}")
    raise ValueError(
        f"Neither {branches[0]} nor {branches[1]} branches could be found in the repository."
    )


def shallow_clone_repository(repo_url: str, repo_parent_path: Path):
    repo_name = repo_url.split("/")[-1].split(".")[0]
    repo_path = f"{repo_parent_path}/{repo_name}"
    if not os.path.exists(repo_path):
        try_clone_repo(repo_url=repo_url, repo_path=repo_path)
    return repo_path


def read_document(document_path: str):
    with open(document_path, "r") as file:
        try:
            document = file.read()
        except UnicodeDecodeError:
            document = ""
    return document


def chunk_document(document: Any):
    parser = TreeSitterPythonParser(document=document)
    chunks = parser.create_chunks()
    main_code = parser.extract_main_code()
    import_statements = parser.extract_import_statements()
    chunks_df = pd.DataFrame(chunks)
    new_rows = pd.DataFrame(
        [
            {"code": main_code, "type": "main_code"},
            {"code": import_statements, "type": "imports"},
        ]
    )
    chunks_df = pd.concat([chunks_df, new_rows], ignore_index=True)
    chunks_df = chunks_df[chunks_df["code"].apply(lambda x: len(x)) != 0]
    return chunks_df


def read_and_chunk_all_python_files(directory_path: Path):
    all_chunks_df = pd.DataFrame()
    python_files = glob.glob(f"{directory_path}/**/*.py", recursive=True)
    for file_path in python_files:
        document = read_document(file_path)
        chunks_df = chunk_document(document)
        chunks_df["file_path"] = file_path
        all_chunks_df = pd.concat([all_chunks_df, chunks_df], ignore_index=True)
    return all_chunks_df


def num_tokens(text: str, model: str) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    print(text)
    return len(encoding.encode(text))


def parse_embedding(embedding: str | list[str]):
    if isinstance(embedding, str):
        return json.loads(embedding)
    elif isinstance(embedding, list):
        return embedding
    else:
        raise ValueError(f"Unexpected value: {embedding}")


def create_openai_embeddings(
    inputs: list[str], embedding_model_name: str, batch_size=10
):
    embeddings: list[list[float]] = []
    for batch_start in range(0, len(inputs), batch_size):
        batch_end = batch_start + batch_size
        batch = inputs[batch_start:batch_end]
        batch = [
            item[:1000] if num_tokens(item, embedding_model_name) > 8192 else item
            for item in batch
        ]
        print(f"Batch {batch_start} to {batch_end-1}")
        EmbeddingResponseData = TypedDict(
            "EmbeddingResponseData",
            {"index": int, "object": str, "embedding": list[float]},
        )
        EmbeddingResponse = TypedDict(
            "EmbeddingResponse", {"data": list[EmbeddingResponseData]}
        )
        response = cast(
            EmbeddingResponse,
            openai.Embedding.create(model=embedding_model_name, input=batch),
        )
        for i, be in enumerate(response["data"]):
            assert (
                i == be["index"]
            )  # double check embeddings are in same order as input
        batch_embeddings = [e["embedding"] for e in response["data"]]
        embeddings.extend(batch_embeddings)
    return embeddings


def create_query_embedding(query: str, embedding_model_name: str):
    embeddings = create_openai_embeddings([query], embedding_model_name)
    query_embedding: FloatNDArray = np.array(embeddings[0]).astype(float).reshape(1, -1)
    return query_embedding


def compute_cosine_similarity(
    chunks_embeddings: FloatNDArray, query_embedding: FloatNDArray
):
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


def ask_gpt(query: str, context: str):
    message = create_message(query, context)
    messages = [
        {"role": "system", "content": "You are an expert programmer"},
        {"role": "user", "content": message},
    ]
    model = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0, stream=True
    )
    for chunk in response:
        yield json.dumps(chunk)


class InferencePipeline:
    def __init__(
        self,
        repo_url: str,
        repo_parent_path: str | None = None,
        start_index_folder_path: str | None = None,
    ):
        self.repo_url = repo_url
        self.repo_parent_path = (
            Path(repo_parent_path) if repo_parent_path else Path.cwd()
        )
        self.start_index_folder_path = (
            Path(start_index_folder_path) if start_index_folder_path else None
        )
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

    def clone_and_process_repo(self):
        repo_name = self.repo_url.split("/")[-1].split(".")[0]
        if os.path.exists(self.repo_parent_path / f"{repo_name}.csv"):
            self.corpus_df = pd.read_csv(self.repo_parent_path / f"{repo_name}.csv")
        else:
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

        if not os.path.exists(self.repo_parent_path / f"{repo_name}.csv"):
            repo_path = shallow_clone_repository(
                repo_url=self.repo_url, repo_parent_path=self.repo_parent_path
            )
            repo_path = Path(repo_path)
            directory_path = (
                self.repo_parent_path / self.start_index_folder_path
                if self.start_index_folder_path
                else repo_path
            )
            all_chunks_df = read_and_chunk_all_python_files(
                directory_path=directory_path
            )
            chosen_types = ["class_definition", "function_definition"]
            chunks_with_embeddings_df = all_chunks_df[
                all_chunks_df["type"].isin(chosen_types)
            ]
            chunks_without_embeddings_df = all_chunks_df[
                ~all_chunks_df["type"].isin(chosen_types)
            ]
            chunks_with_embeddings_df.loc[:, "embedding"] = create_openai_embeddings(
                chunks_with_embeddings_df["code"].tolist(),
                "text-embedding-ada-002",
                batch_size=100,
            )
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
            self.corpus_df.to_csv(
                self.repo_parent_path / f"{repo_name}.csv", index=False
            )

    def compute_similarities(self, query: str):
        chosen_types = ["class_definition", "function_definition"]
        chunks_with_embeddings_df = self.corpus_df[
            self.corpus_df["type"].isin(chosen_types)
        ]
        chunks_with_import_statements = self.corpus_df[
            self.corpus_df["type"].isin(["import_statement", "import_from_statement"])
        ]
        chunk_embeddings = np.array(
            chunks_with_embeddings_df["embedding"].apply(parse_embedding).tolist()
        ).astype(float)
        query_embedding = create_query_embedding(query, "text-embedding-ada-002")
        # similarities = compute_cosine_similarity(chunk_embeddings, query_embedding)
        chunks = chunks_with_embeddings_df["code"].tolist()
        top_chunks, _ = get_top_chunks(
            chunks, chunk_embeddings, query_embedding, top_n=3
        )
        top_chunk_with_imports = add_imports_to_code(
            imports=chunks_with_import_statements["code"].tolist(), code=top_chunks[0]
        )
        return top_chunks, top_chunk_with_imports

    def get_response(self, query: str):
        top_chunks, _ = self.compute_similarities(query=query)
        for sample_response in ask_gpt(query, context="\n".join(top_chunks[:2])):
            yield sample_response


if __name__ == "__main__":
    pipeline = InferencePipeline(
        repo_url="https://github.com/langchain-ai/langchain.git",
        repo_parent_path="samples",
        start_index_folder_path="langchain/libs/langchain/langchain/document_transformers",
    )
    pipeline.clone_and_process_repo()
    query = "Write python function to read a HTML file and transform it into text using Langchain"
    response = pipeline.get_response(query=query)
