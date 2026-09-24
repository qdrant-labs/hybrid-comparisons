import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Any, TypedDict

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_pre_embedded")

LOG_EVERY = 10_000

DATASET_REPO = "CohereLabs/beir-embed-english-v3"

# Only datasets small enough that loading their embeddings isn't itself the
# bottleneck (< ~1M corpus rows).
BEIR_DATASETS = [
    "trec-covid",
    "scidocs",
    "cqadupstack-unix",
    "cqadupstack-gaming",
    "cqadupstack-android",
]


class Passage(TypedDict):
    _id: str
    title: str
    text: str
    emb: list[float]


class Query(TypedDict):
    _id: str
    text: str
    emb: list[float]


class Qrel(TypedDict):
    query_id: str
    corpus_id: str
    score: float


def _write_jsonl_gz(path: str, rows: dict[str, Any]) -> int:
    written = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            written += 1
            if written % LOG_EVERY == 0:
                logger.info("Wrote %d rows to %s", written, path)
    return written


class BeirDataDownloader:
    def download_corpus(self, dataset: str) -> list[Passage]:
        ds = load_dataset(DATASET_REPO, f"{dataset}-corpus", split="train")
        return [
            {
                "_id": row["_id"],
                "title": row["title"],
                "text": row["text"],
                "emb": row["emb"],
            }
            for row in ds
        ]

    def download_queries(self, dataset: str) -> list[Query]:
        ds = load_dataset(DATASET_REPO, f"{dataset}-queries", split="test")
        return [
            {"_id": row["_id"], "text": row["text"], "emb": row["emb"]} for row in ds
        ]

    def download_qrels(self, dataset: str) -> list[Qrel]:
        ds = load_dataset(DATASET_REPO, f"{dataset}-qrels", split="test")
        return [
            {
                "query_id": row["query_id"],
                "corpus_id": row["corpus_id"],
                "score": row["score"],
            }
            for row in ds
        ]

    def download(
        self, dataset: str, out_corpus: str, out_queries: str, out_qrels: str
    ) -> None:
        logger.info("[%s] downloading corpus", dataset)
        corpus = self.download_corpus(dataset)
        logger.info("[%s] downloaded %d corpus rows", dataset, len(corpus))
        n = _write_jsonl_gz(out_corpus, corpus)
        logger.info("[%s] wrote %d corpus rows to %s", dataset, n, out_corpus)

        logger.info("[%s] downloading queries", dataset)
        queries = self.download_queries(dataset)
        logger.info("[%s] downloaded %d queries", dataset, len(queries))
        n = _write_jsonl_gz(out_queries, queries)
        logger.info("[%s] wrote %d queries to %s", dataset, n, out_queries)

        logger.info("[%s] downloading qrels", dataset)
        qrels = self.download_qrels(dataset)
        logger.info("[%s] downloaded %d qrels", dataset, len(qrels))
        n = _write_jsonl_gz(out_qrels, qrels)
        logger.info("[%s] wrote %d qrels to %s", dataset, n, out_qrels)


def main() -> None:
    args = sys.argv

    if len(args) < 2:
        print("Usage: download-pre-embedded <dataset> [out_dir]")
        print(f"Available datasets: {', '.join(BEIR_DATASETS)}")
        sys.exit(2)

    dataset = args[1]
    if dataset not in BEIR_DATASETS:
        print(f"Unsupported dataset: {dataset}. Available: {', '.join(BEIR_DATASETS)}")
        sys.exit(2)

    out_dir = args[2] if len(args) >= 3 else "."

    if not Path(out_dir).exists():
        Path(out_dir).mkdir(exist_ok=True, parents=True)

    downloader = BeirDataDownloader()
    downloader.download(
        dataset,
        out_corpus=f"{out_dir}/{dataset}-corpus.jsonl.gz",
        out_queries=f"{out_dir}/{dataset}-queries.jsonl.gz",
        out_qrels=f"{out_dir}/{dataset}-qrels.jsonl.gz",
    )


if __name__ == "__main__":
    main()
