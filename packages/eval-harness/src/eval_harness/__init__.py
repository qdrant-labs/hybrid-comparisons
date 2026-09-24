# pyright: basic

import argparse
import asyncio
import gzip
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple, TypedDict, cast

import aiofiles
import httpx
from qdrant_client import AsyncQdrantClient, models
from qdrant_load import (
    COLBERT_MODEL,
    DENSE_MODEL,
    SPARSE_MODEL,
    get_qdrant_client,
    load_config,
    load_corpus,
    load_points,
    with_retries,
)

# Local reranking alternative to the server-side colbert rescore: small,
# fastembed-native cross-encoder, run on the raw text of the prefetched
# candidates.
CROSS_ENCODER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

# Doc ids are ints for the synthetic (non-pre-embedded) corpus and strings
# (BEIR's original _id/corpus_id) for pre-embedded corpora.
DocId = str | int


class EvalQuery(NamedTuple):
    qid: str
    text: str
    # Precomputed query embedding for pre-embedded corpora (must be used for
    # the dense stage instead of Qdrant Cloud Inference's own dense model,
    # since the corpus was embedded with Cohere, not all-minilm). None for
    # the synthetic corpus, where the dense stage embeds `text` server-side.
    emb: list[float] | None


class RerankRequest(TypedDict):
    query: str
    documents: list[str]
    return_documents: bool


class RerankResponseItem(TypedDict):
    index: int
    score: float
    document: str | None

class RerankResponse(TypedDict):
    items: list[RerankResponseItem]


async def rerank(query: str, documents: list[str]) -> list[float]:
    base_url = get_cross_encoder_endpoint()
    async with httpx.AsyncClient(base_url=base_url, timeout=600) as client:
        response = await client.post(
            "/rerank", json=RerankRequest(query=query, documents=documents, return_documents=False)
        )
        response.raise_for_status()
        data: RerankResponse = response.json()
        return [x["score"] for x in data["items"]]


@lru_cache(maxsize=1)
def get_cross_encoder_endpoint() -> str:
    endpoint = os.getenv("CROSS_ENCODER_ENDPOINT")
    if endpoint is None:
        raise RuntimeError(
            "Could not find CROSS_ENCODER_ENDPOINT in the current environment"
        )
    return endpoint


def load_queries(
    corpus_path: str, n_queries: int | None, seed: int
) -> tuple[list[EvalQuery], dict[str, set[DocId]]]:
    """Builds an eval query set straight from the corpus's real, associated
    search queries (no synthetic/LLM generation, no separate query dir):
    for each sampled passage with at least one real query, pick one of its
    queries and treat that same passage as the single ground-truth relevant
    doc.

    n_queries=None uses every passage that has an associated query.
    """
    corpus = load_corpus(corpus_path, False)
    candidates = [row for row in corpus if row["queries"]]
    if n_queries is None:
        n_queries = len(candidates)
    if n_queries > len(candidates):
        raise RuntimeError(
            f"Only {len(candidates)} passages have associated queries, cannot sample {n_queries}"
        )
    rng = random.Random(seed)
    sampled = rng.sample(candidates, n_queries)

    queries: list[EvalQuery] = []
    qrels: dict[str, set[DocId]] = {}
    for row in sampled:
        qid = f"q{row['pid']}"
        query_text = rng.choice(row["queries"])
        queries.append(EvalQuery(qid, query_text, None))
        qrels[qid] = {row["pid"]}
    return queries, qrels


def _read_jsonl_gz(path: str) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _default_sibling_path(corpus_path: str, kind: str) -> str:
    """Derives the queries/qrels path from a pre-embedded corpus path, e.g.
    '.../trec-covid-corpus.jsonl.gz' -> '.../trec-covid-queries.jsonl.gz',
    following the naming convention download-pre-embedded writes."""
    suffix = "-corpus.jsonl.gz"
    if not corpus_path.endswith(suffix):
        raise RuntimeError(
            f"Cannot infer default {kind} path from {corpus_path!r}; "
            f"pass --{kind}-path explicitly"
        )
    return corpus_path[: -len(suffix)] + f"-{kind}.jsonl.gz"


def load_pre_embedded_queries(
    queries_path: str,
    qrels_path: str,
    n_queries: int | None,
    seed: int,
    relevance_threshold: float = 0.0,
) -> tuple[list[EvalQuery], dict[str, set[DocId]]]:
    """Loads the real queries and NIST/human qrels shipped by
    download-pre-embedded for BEIR corpora, instead of the synthetic
    single-doc ground truth `load_queries` derives from the corpus itself.

    Query text is still sent for the sparse/colbert stages, but each query's
    own precomputed embedding is used for the dense stage so it lands in the
    same space as the corpus's Cohere embeddings -- embedding the query text
    server-side with Qdrant Cloud Inference's default dense model would
    search the right vectors with the wrong model entirely.

    n_queries=None uses every query that has at least one qrel above the
    relevance threshold.
    """
    all_queries = _read_jsonl_gz(queries_path)
    qrels_rows = _read_jsonl_gz(qrels_path)

    qrels: dict[str, set[DocId]] = {}
    for row in qrels_rows:
        if row["score"] > relevance_threshold:
            qrels.setdefault(row["query_id"], set()).add(row["corpus_id"])

    eligible = [q for q in all_queries if qrels.get(q["_id"])]
    if n_queries is None:
        n_queries = len(eligible)
    if n_queries > len(eligible):
        raise RuntimeError(
            f"Only {len(eligible)} queries have at least one qrel above the relevance "
            f"threshold ({relevance_threshold}), cannot sample {n_queries}"
        )
    rng = random.Random(seed)
    sampled = rng.sample(eligible, n_queries)

    queries = [EvalQuery(q["_id"], q["text"], q["emb"]) for q in sampled]
    sampled_ids = {q["_id"] for q in sampled}
    return queries, {qid: ids for qid, ids in qrels.items() if qid in sampled_ids}


def _pid(point: models.ScoredPoint) -> DocId:
    """Always resolve a result's doc id from its payload, not point.id: for
    pre-embedded collections point.id is a uuid5 derived from the original
    BEIR _id (Qdrant point ids must be an int or a UUID), so the original id
    -- the one qrels actually reference -- only lives in payload['pid']."""
    return cast(dict[str, Any], point.payload)["pid"]


async def search(
    client: AsyncQdrantClient,
    collection_name: str,
    query_text: str,
    k: int,
    prefetch_limit: int,
    use_dense_prefetch: bool = True,
    rescorer: str = "colbert",
    query_emb: list[float] | None = None,
) -> list[DocId]:
    dense_query = (
        query_emb
        if query_emb is not None
        else models.Document(text=query_text, model=DENSE_MODEL)
    )
    prefetch_stages = (
        [
            models.Prefetch(
                query=dense_query,
                using="dense",
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=models.Document(text=query_text, model=SPARSE_MODEL),
                using="sparse",
                limit=prefetch_limit,
            ),
        ]
        if use_dense_prefetch
        else [
            models.Prefetch(
                query=models.Document(text=query_text, model=SPARSE_MODEL),
                using="sparse",
                limit=prefetch_limit,
            ),
        ]
    )

    if rescorer == "rrf":
        # no second-stage rescore at all: dense + sparse prefetch, fused by
        # RRF, *is* the final result.
        if not use_dense_prefetch:
            raise ValueError(
                "rescorer='rrf' needs use_dense_prefetch=True (RRF fuses >=2 sources)"
            )
        result = await with_retries(
            client.query_points,
            collection_name=collection_name,
            prefetch=prefetch_stages,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=["pid"],
        )
        return [_pid(point) for point in result.points]

    if rescorer == "colbert":
        prefetch = (
            models.Prefetch(
                prefetch=prefetch_stages,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=prefetch_limit,
            )
            if use_dense_prefetch
            else prefetch_stages[0]
        )
        result = await with_retries(
            client.query_points,
            collection_name=collection_name,
            prefetch=prefetch,
            query=models.Document(text=query_text, model=COLBERT_MODEL),
            using="colbert",
            limit=k,
            with_payload=["pid"],
        )
        return [_pid(point) for point in result.points]

    # rescorer == "cross-encoder": the prefetch stage(s) *are* the top-level
    # query -- retrieve prefetch_limit candidates with their text, then
    # rerank locally instead of a second server-side vector stage.
    if use_dense_prefetch:
        result = await with_retries(
            client.query_points,
            collection_name=collection_name,
            prefetch=prefetch_stages,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=prefetch_limit,
            with_payload=["pid", "text"],
        )
    else:
        result = await with_retries(
            client.query_points,
            collection_name=collection_name,
            query=models.Document(text=query_text, model=SPARSE_MODEL),
            using="sparse",
            limit=prefetch_limit,
            with_payload=["pid", "text"],
        )
    candidates = [
        (_pid(p), cast(dict[str, Any], p.payload)["text"]) for p in result.points
    ]
    if not candidates:
        return []
    scores = await with_retries(
        rerank,
        query_text,
        [text for _, text in candidates],
    )
    ranked = sorted(
        zip(scores, (pid for pid, _ in candidates)), key=lambda x: x[0], reverse=True
    )
    return [pid for _, pid in ranked[:k]]


def compute_quality(
    results: list[tuple[str, list[DocId]]], qrels: dict[str, set[DocId]]
) -> dict[str, float]:
    """`qrels` maps a query id to its set of ground-truth relevant doc ids.
    Relevance is treated as binary.
    """
    n = len(results)
    hit_count = 0
    recall_sum = 0.0
    rr_sum = 0.0
    ndcg_sum = 0.0
    for qid, ids in results:
        relevant = qrels[qid]
        if not relevant:
            continue
        found_ranks = [i + 1 for i, doc_id in enumerate(ids) if doc_id in relevant]
        recall_sum += len(found_ranks) / len(relevant)
        if found_ranks:
            hit_count += 1
            rr_sum += 1.0 / found_ranks[0]
        dcg = sum(1.0 / math.log2(rank + 1) for rank in found_ranks)
        ideal_hits = min(len(relevant), len(ids))
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
        if idcg > 0:
            ndcg_sum += dcg / idcg
    return {
        "recall@k": recall_sum / n,
        "hit_rate@k": hit_count / n,
        "mrr@k": rr_sum / n,
        "ndcg@k": ndcg_sum / n,
    }


def latency_stats(latencies: list[float]) -> dict[str, float]:
    sorted_lat = sorted(latencies)
    return {
        "mean_ms": statistics.mean(sorted_lat) * 1000,
        "p50_ms": sorted_lat[int(len(sorted_lat) * 0.50)] * 1000,
        "p95_ms": sorted_lat[min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)]
        * 1000,
        "p99_ms": sorted_lat[min(int(len(sorted_lat) * 0.99), len(sorted_lat) - 1)]
        * 1000,
        "min_ms": sorted_lat[0] * 1000,
        "max_ms": sorted_lat[-1] * 1000,
    }


async def bench_latency(
    client: AsyncQdrantClient,
    collection_name: str,
    queries: list[EvalQuery],
    k: int,
    prefetch_limit: int,
    warmup: int,
    use_dense_prefetch: bool,
    rescorer: str,
) -> tuple[list[tuple[str, list[DocId]]], list[float]]:
    results: list[tuple[str, list[DocId]]] = []
    latencies: list[float] = []
    for i, (qid, query_text, query_emb) in enumerate(queries):
        start = time.perf_counter()
        ids = await search(
            client,
            collection_name,
            query_text,
            k,
            prefetch_limit,
            use_dense_prefetch,
            rescorer,
            query_emb=query_emb,
        )
        elapsed = time.perf_counter() - start
        if i >= warmup:
            latencies.append(elapsed)
        results.append((qid, ids))
    return results, latencies


async def bench_throughput(
    client: AsyncQdrantClient,
    collection_name: str,
    queries: list[EvalQuery],
    k: int,
    prefetch_limit: int,
    concurrency: int,
    use_dense_prefetch: bool,
    rescorer: str,
) -> float:
    semaphore = asyncio.Semaphore(concurrency)

    async def bound(q: EvalQuery) -> list[DocId]:
        _, query_text, query_emb = q
        async with semaphore:
            return await search(
                client,
                collection_name,
                query_text,
                k,
                prefetch_limit,
                use_dense_prefetch,
                rescorer,
                query_emb=query_emb,
            )

    start = time.perf_counter()
    await asyncio.gather(*(bound(q) for q in queries))
    elapsed = time.perf_counter() - start
    return len(queries) / elapsed


RESCORER_TAGS = {"colbert": "colbert", "cross-encoder": "ce", "rrf": "rrf"}


def report_filename(
    collection_name: str,
    k: int,
    prefetch_limit: int,
    use_dense_prefetch: bool,
    rescorer: str,
) -> str:
    prefetch_tag = "hybrid" if use_dense_prefetch else "sparseonly"
    rescorer_tag = RESCORER_TAGS[rescorer]
    return (
        f"{collection_name}_k{k}_pf{prefetch_limit}_{prefetch_tag}_{rescorer_tag}.json"
    )


def report_path(out_dir: Path, report: dict[str, Any]) -> Path:
    return out_dir / report_filename(
        report["collection_name"],
        report["k"],
        report["prefetch_limit"],
        report["use_dense_prefetch"],
        report["rescorer"],
    )


def existing_report_paths(
    out_dir: Path,
    collection_name: str,
    k: int,
    prefetch_limit: int,
    use_dense_prefetch: bool,
    rescorer: str,
) -> list[Path]:
    """Every filename a report for this combination could have been written
    under -- including the legacy naming (no rescorer suffix) from before
    multiple rescorers existed, back when every run was implicitly colbert.
    """
    paths = [
        out_dir
        / report_filename(
            collection_name, k, prefetch_limit, use_dense_prefetch, rescorer
        )
    ]
    if rescorer == "colbert":
        prefetch_tag = "hybrid" if use_dense_prefetch else "sparseonly"
        paths.append(
            out_dir / f"{collection_name}_k{k}_pf{prefetch_limit}_{prefetch_tag}.json"
        )
    return paths


async def run_sweep(
    config_path: str,
    n_queries: int | None,
    seed: int,
    k_values: list[int],
    prefetch_values: list[int],
    modes: list[bool],
    rescorers: list[str],
    warmup: int,
    concurrency: int,
    output_dir: str,
    overwrite: bool = False,
    fresh: bool = False,
    pre_embedded: bool = False,
    queries_path: str | None = None,
    qrels_path: str | None = None,
) -> list[dict[str, Any]]:
    """Uploads the collection for `config_path` exactly once, then runs every
    (mode, rescorer, k, prefetch_limit) combination against it -- so a sweep
    across search settings never re-uploads or re-embeds the corpus.

    When `pre_embedded` is set, queries and ground truth come from the real
    queries/qrels files download-pre-embedded wrote alongside the corpus
    (defaulting to the sibling `*-queries.jsonl.gz` / `*-qrels.jsonl.gz` next
    to the corpus file) instead of the synthetic single-doc ground truth
    `load_queries` derives from the corpus itself.

    Resumable by default: a combination whose report file already exists in
    `output_dir` is skipped (pass `overwrite=True` to redo it anyway), and if
    the collection already exists (e.g. left behind by an interrupted prior
    run of this same config) it's reused rather than re-uploaded (pass
    `fresh=True` to force a clean delete + re-upload instead).
    """
    cfg = load_config(config_path)
    if pre_embedded != cfg.pre_embedded:
        raise RuntimeError(
            f"--pre-embedded={pre_embedded} does not match {config_path}'s "
            f"pre_embedded={cfg.pre_embedded}"
        )
    client = get_qdrant_client()
    already_exists = await with_retries(
        client.collection_exists, collection_name=cfg.collection_name
    )
    if already_exists and fresh:
        print(
            f"--fresh: deleting existing collection {cfg.collection_name} before re-uploading"
        )
        await with_retries(
            client.delete_collection, collection_name=cfg.collection_name
        )
        already_exists = False
    if already_exists:
        print(
            f"Collection {cfg.collection_name} already exists -- reusing it (no re-upload)"
        )
    else:
        await load_points(config_path)
    try:
        if pre_embedded:
            qp = queries_path or _default_sibling_path(cfg.data.corpus.path, "queries")
            rp = qrels_path or _default_sibling_path(cfg.data.corpus.path, "qrels")
            queries, qrels = load_pre_embedded_queries(qp, rp, n_queries, seed)
        else:
            queries, qrels = load_queries(cfg.data.corpus.path, n_queries, seed)
        if warmup >= len(queries):
            raise RuntimeError(
                f"warmup ({warmup}) must be smaller than the number of queries ({len(queries)})"
            )

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        reports: list[dict[str, Any]] = []
        for use_dense_prefetch in modes:
            for rescorer in rescorers:
                if rescorer == "rrf" and not use_dense_prefetch:
                    print(
                        "skip: mode=sparseonly rescorer=rrf (rrf needs dense+sparse to fuse)"
                    )
                    continue
                for k in k_values:
                    for prefetch_limit in prefetch_values:
                        if prefetch_limit < k:
                            print(
                                f"skip: mode={'hybrid' if use_dense_prefetch else 'sparseonly'} "
                                f"rescorer={rescorer} k={k} prefetch_limit={prefetch_limit} "
                                "(prefetch_limit < k)"
                            )
                            continue

                        if not overwrite:
                            done = [
                                p
                                for p in existing_report_paths(
                                    out_dir,
                                    cfg.collection_name,
                                    k,
                                    prefetch_limit,
                                    use_dense_prefetch,
                                    rescorer,
                                )
                                if p.exists()
                            ]
                            if done:
                                print(
                                    f"skip: mode={'hybrid' if use_dense_prefetch else 'sparseonly'} "
                                    f"rescorer={rescorer} k={k} prefetch_limit={prefetch_limit} "
                                    f"(already have {done[0].name})"
                                )
                                continue

                        print(
                            f"\n=== mode={'hybrid' if use_dense_prefetch else 'sparseonly'} "
                            f"rescorer={rescorer} k={k} prefetch_limit={prefetch_limit} ==="
                        )
                        results, latencies = await bench_latency(
                            client,
                            cfg.collection_name,
                            queries,
                            k,
                            prefetch_limit,
                            warmup,
                            use_dense_prefetch,
                            rescorer,
                        )
                        quality = compute_quality(results, qrels)
                        latency = latency_stats(latencies)
                        qps = await bench_throughput(
                            client,
                            cfg.collection_name,
                            queries,
                            k,
                            prefetch_limit,
                            concurrency,
                            use_dense_prefetch,
                            rescorer,
                        )

                        report = {
                            "collection_name": cfg.collection_name,
                            "config": config_path,
                            "n_queries": len(queries),
                            "k": k,
                            "prefetch_limit": prefetch_limit,
                            "use_dense_prefetch": use_dense_prefetch,
                            "rescorer": rescorer,
                            "concurrency": concurrency,
                            "quality": quality,
                            "latency": latency,
                            "throughput_qps": qps,
                        }
                        print_report(report)
                        out_path = report_path(out_dir, report)
                        async with aiofiles.open(out_path, "w") as f:
                            rep = json.dumps(report, indent=2)
                            await f.write(rep)
                        print(f"Wrote report to {out_path}")
                        reports.append(report)
    finally:
        await with_retries(
            client.delete_collection, collection_name=cfg.collection_name
        )
    return reports


def print_report(report: dict[str, Any]) -> None:
    print(f"\n=== {report['collection_name']} ({report['config']}) ===")
    print(
        f"queries: {report['n_queries']}  k: {report['k']}  prefetch_limit: {report['prefetch_limit']}  "
        f"use_dense_prefetch: {report['use_dense_prefetch']}  rescorer: {report['rescorer']}"
    )
    print("-- quality --")
    for name, value in report["quality"].items():
        print(f"  {name}: {value:.4f}")
    print("-- latency (sequential) --")
    for name, value in report["latency"].items():
        print(f"  {name}: {value:.2f}")
    print(f"-- throughput ({report['concurrency']} concurrent) --")
    print(f"  qps: {report['throughput_qps']:.2f}")


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_mode_list(s: str) -> list[bool]:
    modes = [x.strip() for x in s.split(",") if x.strip()]
    result = []
    for m in modes:
        if m == "hybrid":
            result.append(True)
        elif m == "sparseonly":
            result.append(False)
        else:
            raise ValueError(
                f"Unknown prefetch mode: {m!r} (expected 'hybrid' or 'sparseonly')"
            )
    return result


def parse_rescorer_list(s: str) -> list[str]:
    rescorers = [x.strip() for x in s.split(",") if x.strip()]
    for r in rescorers:
        if r not in ("colbert", "cross-encoder", "rrf"):
            raise ValueError(
                f"Unknown rescorer: {r!r} (expected colbert, cross-encoder or rrf)"
            )
    return rescorers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate search quality and speed for a loaded collection, sweeping "
        "search settings against a single upload of the corpus"
    )
    parser.add_argument(
        "config", help="path to the qdrant-load config.yml used to load the collection"
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=None,
        help="number of real corpus queries to sample for evaluation "
        "(default: use every available eligible query)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--k", default="3", help="comma-separated list of k values, e.g. 10,20"
    )
    parser.add_argument(
        "--prefetch-limit",
        default="5",
        help="comma-separated list of prefetch_limit values",
    )
    parser.add_argument(
        "--modes",
        default="hybrid",
        help="comma-separated list of prefetch modes: hybrid, sparseonly",
    )
    parser.add_argument(
        "--rescorers",
        default="colbert",
        help="comma-separated list of final rescoring stages: colbert (server-side MaxSim), "
        "cross-encoder (remote rerank via CROSS_ENCODER_ENDPOINT), rrf (no second stage; needs hybrid mode)",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-dir", default="../../results")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="re-run and overwrite combinations that already have a report in --output-dir "
        "(default: skip them, so an interrupted sweep can just be re-run to resume)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help="delete and re-upload the collection even if it already exists "
        "(default: reuse an existing collection, e.g. left behind by an interrupted run)",
    )
    parser.add_argument(
        "--pre-embedded",
        action="store_true",
        default=False,
        help="use the real queries/qrels shipped by download-pre-embedded (BEIR corpora with "
        "precomputed Cohere embeddings) instead of synthetic single-doc ground truth derived "
        "from the corpus; must match the config's own pre_embedded setting",
    )
    parser.add_argument(
        "--queries-path",
        default=None,
        help="path to the *-queries.jsonl.gz file (only used with --pre-embedded; "
        "defaults to the file sitting next to the corpus)",
    )
    parser.add_argument(
        "--qrels-path",
        default=None,
        help="path to the *-qrels.jsonl.gz file (only used with --pre-embedded; "
        "defaults to the file sitting next to the corpus)",
    )
    args = parser.parse_args()

    try:
        k_values = parse_int_list(args.k)
        prefetch_values = parse_int_list(args.prefetch_limit)
        modes = parse_mode_list(args.modes)
        rescorers = parse_rescorer_list(args.rescorers)
    except ValueError as e:
        parser.error(str(e))

    asyncio.run(
        run_sweep(
            args.config,
            args.n_queries,
            args.seed,
            k_values,
            prefetch_values,
            modes,
            rescorers,
            args.warmup,
            args.concurrency,
            args.output_dir,
            args.overwrite,
            args.fresh,
            args.pre_embedded,
            args.queries_path,
            args.qrels_path,
        )
    )


if __name__ == "__main__":
    main()
