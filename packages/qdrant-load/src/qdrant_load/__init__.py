# pyright: basic
"""Uploads a corpus (see download-data's corpus.jsonl) to a Qdrant Cloud
collection using Qdrant Cloud Inference: dense, sparse and ColBERT vectors
are all computed server-side from raw text (`models.Document`), so no local
embedding step is needed at all.
"""

import asyncio
import json
import os
import random
import sys
from functools import lru_cache
from typing import Awaitable, Callable, Literal, TypedDict, TypeVar

import grpc
import httpx
import yaml
from pydantic import BaseModel, ConfigDict
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

DENSE_MODEL = "sentence-transformers/all-minilm-l6-v2"
SPARSE_MODEL = "Qdrant/bm25"
COLBERT_MODEL = "answerdotai/answerai-colbert-small-v1"

DENSE_SIZE = 384
COLBERT_SIZE = 96

UPLOAD_BATCH_SIZE = 500

# Transient network/connection failures worth retrying -- not validation
# errors or other permanent 4xx failures, which should surface immediately.
RETRYABLE_EXCEPTIONS = (
    ResponseHandlingException,
    httpx.TransportError,
    httpx.TimeoutException,
    grpc.aio.AioRpcError,
    ConnectionError,
    TimeoutError,
    OSError,
)

T = TypeVar("T")


async def with_retries[T](
    fn: Callable[..., Awaitable[T]],
    *args: object,
    retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    **kwargs: object,
) -> T:
    """Retries `fn` with exponential backoff + jitter on transient network
    failures (dropped connections, timeouts, DNS blips) -- the kind of thing
    that otherwise kills an unattended overnight sweep on a single hiccup.
    Anything else (bad request, auth failure, ...) propagates immediately.
    """
    for attempt in range(retries + 1):
        try:
            return await fn(*args, **kwargs)
        except RETRYABLE_EXCEPTIONS as e:
            if attempt == retries:
                raise
            delay = min(max_delay, base_delay * (2**attempt)) * (0.5 + random.random())
            print(
                f"[retry] {type(e).__name__}: {e} -- retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{retries})"
            )
            await asyncio.sleep(delay)
    raise AssertionError("Cannot be reached")

class DenseUploadConfig(BaseModel):
    dtype: models.Datatype | None = None
    quantization_config: models.QuantizationConfig | None = None
    multivector_config: models.MultiVectorConfig | None = None
    hnsw_config: models.HnswConfigDiff | None = None
    memory: models.Memory | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SparseUploadConfig(BaseModel):
    modifier: models.Modifier | None = None
    index_config: models.SparseIndexParams | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CorpusDataConfig(BaseModel):
    path: str
    type: Literal["corpus"] = "corpus"


class DataConfig(BaseModel):
    corpus: CorpusDataConfig


class UploadConfig(BaseModel):
    collection_name: str
    data: DataConfig
    sparse_vectors: SparseUploadConfig = SparseUploadConfig()
    dense_vectors: DenseUploadConfig = DenseUploadConfig()
    colbert_vectors: DenseUploadConfig = DenseUploadConfig()


class CorpusRow(TypedDict):
    pid: int
    text: str
    queries: list[str]


def load_corpus(path: str) -> list[CorpusRow]:
    """Loads the JSONL corpus written by download-data: one
    {"pid", "text", "queries"} object per line.
    """
    rows: list[CorpusRow] = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def to_point(row: CorpusRow) -> models.PointStruct:
    return models.PointStruct(
        id=row["pid"],
        vector={
            "dense": models.Document(text=row["text"], model=DENSE_MODEL),
            "sparse": models.Document(text=row["text"], model=SPARSE_MODEL),
            "colbert": models.Document(text=row["text"], model=COLBERT_MODEL),
        },
        payload={"pid": row["pid"], "text": row["text"]},
    )


@lru_cache(maxsize=1)
def get_qdrant_client() -> AsyncQdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if url is None:
        raise RuntimeError("QDRANT_URL not found in the current environment")
    return AsyncQdrantClient(
        url=url,
        api_key=api_key,
        cloud_inference=True,
        prefer_grpc=True,
        timeout=120,
    )


def load_config(path: str) -> UploadConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    config = UploadConfig.model_validate(d)
    return config


async def load_points(config_path: str) -> None:
    cfg = load_config(config_path)
    client = get_qdrant_client()
    exists = await with_retries(client.collection_exists, collection_name=cfg.collection_name)
    if exists:
        raise RuntimeError(
            f"Collection {cfg.collection_name} already exists. Rename it or delete it before running this commad again"
        )
    await with_retries(
        client.create_collection,
        collection_name=cfg.collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_SIZE,
                distance=models.Distance.COSINE,
                hnsw_config=cfg.dense_vectors.hnsw_config,
                quantization_config=cfg.dense_vectors.quantization_config,
                datatype=cfg.dense_vectors.dtype,
                memory=cfg.dense_vectors.memory,
            ),
            "colbert": models.VectorParams(
                size=COLBERT_SIZE,
                distance=models.Distance.COSINE,
                hnsw_config=cfg.colbert_vectors.hnsw_config,
                quantization_config=cfg.colbert_vectors.quantization_config,
                datatype=cfg.colbert_vectors.dtype,
                memory=cfg.colbert_vectors.memory,
                multivector_config=cfg.colbert_vectors.multivector_config,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=cfg.sparse_vectors.modifier,
                index=cfg.sparse_vectors.index_config,
            )
        },
    )
    corpus = load_corpus(cfg.data.corpus.path)
    # upload_points already retries internally (network hiccups during a
    # single batch), this just widens that safety margin a bit.
    client.upload_points(
        collection_name=cfg.collection_name,
        points=(to_point(row) for row in corpus),
        batch_size=UPLOAD_BATCH_SIZE,
        max_retries=5,
    )
    print(f"Done uploading {len(corpus)} passages to {cfg.collection_name}!")


def main() -> None:
    args = sys.argv
    config_file = "config.yml"
    if len(args) >= 2:
        config_file = args[1]
    asyncio.run(load_points(config_file))
