# pyright: basic

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from fastembed.rerank.cross_encoder import TextCrossEncoder
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

# (query_id, query text, ground-truth relevant pid)
EvalQuery = tuple[str, str, int]


@lru_cache(maxsize=1)
def get_cross_encoder() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=CROSS_ENCODER_MODEL)


def load_queries(corpus_path: str, n_queries: int, seed: int) -> tuple[list[EvalQuery], dict[str, set[int]]]:
    """Builds an eval query set straight from the corpus's real, associated
    search queries (no synthetic/LLM generation, no separate query dir):
    for each sampled passage with at least one real query, pick one of its
    queries and treat that same passage as the single ground-truth relevant
    doc.
    """
    corpus = load_corpus(corpus_path)
    candidates = [row for row in corpus if row["queries"]]
    if n_queries > len(candidates):
        raise RuntimeError(
            f"Only {len(candidates)} passages have associated queries, cannot sample {n_queries}"
        )
    rng = random.Random(seed)
    sampled = rng.sample(candidates, n_queries)

    queries: list[EvalQuery] = []
    qrels: dict[str, set[int]] = {}
    for row in sampled:
        qid = f"q{row['pid']}"
        query_text = rng.choice(row["queries"])
        queries.append((qid, query_text, row["pid"]))
        qrels[qid] = {row["pid"]}
    return queries, qrels


async def search(
    client: AsyncQdrantClient,
    collection_name: str,
    query_text: str,
    k: int,
    prefetch_limit: int,
    use_dense_prefetch: bool = True,
    rescorer: str = "colbert",
) -> list[int]:
    prefetch_stages = (
        [
            models.Prefetch(
                query=models.Document(text=query_text, model=DENSE_MODEL),
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
            raise ValueError("rescorer='rrf' needs use_dense_prefetch=True (RRF fuses >=2 sources)")
        result = await with_retries(
            client.query_points,
            collection_name=collection_name,
            prefetch=prefetch_stages,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=False,
        )
        return [cast(int, point.id) for point in result.points]

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
            with_payload=False,
        )
        return [cast(int, point.id) for point in result.points]

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
    candidates = [(cast(int, p.id), cast(dict[str, Any], p.payload)["text"]) for p in result.points]
    if not candidates:
        return []
    scores = list(get_cross_encoder().rerank(query_text, [text for _, text in candidates]))
    ranked = sorted(zip(scores, (pid for pid, _ in candidates)), key=lambda x: x[0], reverse=True)
    return [pid for _, pid in ranked[:k]]


def compute_quality(
    results: list[tuple[str, list[int]]], qrels: dict[str, set[int]]
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
) -> tuple[list[tuple[str, list[int]]], list[float]]:
    results: list[tuple[str, list[int]]] = []
    latencies: list[float] = []
    for i, (qid, query_text, _) in enumerate(queries):
        start = time.perf_counter()
        ids = await search(
            client, collection_name, query_text, k, prefetch_limit, use_dense_prefetch, rescorer
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

    async def bound(q: EvalQuery) -> list[int]:
        _, query_text, _ = q
        async with semaphore:
            return await search(
                client, collection_name, query_text, k, prefetch_limit, use_dense_prefetch, rescorer
            )

    start = time.perf_counter()
    await asyncio.gather(*(bound(q) for q in queries))
    elapsed = time.perf_counter() - start
    return len(queries) / elapsed


RESCORER_TAGS = {"colbert": "colbert", "cross-encoder": "ce", "rrf": "rrf"}


def report_filename(
    collection_name: str, k: int, prefetch_limit: int, use_dense_prefetch: bool, rescorer: str
) -> str:
    prefetch_tag = "hybrid" if use_dense_prefetch else "sparseonly"
    rescorer_tag = RESCORER_TAGS[rescorer]
    return f"{collection_name}_k{k}_pf{prefetch_limit}_{prefetch_tag}_{rescorer_tag}.json"


def report_path(out_dir: Path, report: dict[str, Any]) -> Path:
    return out_dir / report_filename(
        report["collection_name"], report["k"], report["prefetch_limit"],
        report["use_dense_prefetch"], report["rescorer"],
    )


def existing_report_paths(
    out_dir: Path, collection_name: str, k: int, prefetch_limit: int, use_dense_prefetch: bool, rescorer: str
) -> list[Path]:
    """Every filename a report for this combination could have been written
    under -- including the legacy naming (no rescorer suffix) from before
    multiple rescorers existed, back when every run was implicitly colbert.
    """
    paths = [out_dir / report_filename(collection_name, k, prefetch_limit, use_dense_prefetch, rescorer)]
    if rescorer == "colbert":
        prefetch_tag = "hybrid" if use_dense_prefetch else "sparseonly"
        paths.append(out_dir / f"{collection_name}_k{k}_pf{prefetch_limit}_{prefetch_tag}.json")
    return paths


async def run_sweep(
    config_path: str,
    n_queries: int,
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
) -> list[dict[str, Any]]:
    """Uploads the collection for `config_path` exactly once, then runs every
    (mode, rescorer, k, prefetch_limit) combination against it -- so a sweep
    across search settings never re-uploads or re-embeds the corpus.

    Resumable by default: a combination whose report file already exists in
    `output_dir` is skipped (pass `overwrite=True` to redo it anyway), and if
    the collection already exists (e.g. left behind by an interrupted prior
    run of this same config) it's reused rather than re-uploaded (pass
    `fresh=True` to force a clean delete + re-upload instead).
    """
    cfg = load_config(config_path)
    client = get_qdrant_client()
    already_exists = await with_retries(client.collection_exists, collection_name=cfg.collection_name)
    if already_exists and fresh:
        print(f"--fresh: deleting existing collection {cfg.collection_name} before re-uploading")
        await with_retries(client.delete_collection, collection_name=cfg.collection_name)
        already_exists = False
    if already_exists:
        print(f"Collection {cfg.collection_name} already exists -- reusing it (no re-upload)")
    else:
        await load_points(config_path)
    try:
        queries, qrels = load_queries(cfg.data.corpus.path, n_queries, seed)
        if warmup >= len(queries):
            raise RuntimeError(
                f"warmup ({warmup}) must be smaller than the number of queries ({len(queries)})"
            )
        if "cross-encoder" in rescorers:
            get_cross_encoder()  # warm the local model load out of the timed path

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        reports: list[dict[str, Any]] = []
        for use_dense_prefetch in modes:
            for rescorer in rescorers:
                if rescorer == "rrf" and not use_dense_prefetch:
                    print("skip: mode=sparseonly rescorer=rrf (rrf needs dense+sparse to fuse)")
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
                                p for p in existing_report_paths(
                                    out_dir, cfg.collection_name, k, prefetch_limit,
                                    use_dense_prefetch, rescorer,
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
                            client, cfg.collection_name, queries, k, prefetch_limit,
                            warmup, use_dense_prefetch, rescorer,
                        )
                        quality = compute_quality(results, qrels)
                        latency = latency_stats(latencies)
                        qps = await bench_throughput(
                            client, cfg.collection_name, queries, k, prefetch_limit,
                            concurrency, use_dense_prefetch, rescorer,
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
                        with open(out_path, "w") as f:
                            json.dump(report, f, indent=2)
                        print(f"Wrote report to {out_path}")
                        reports.append(report)
    finally:
        await with_retries(client.delete_collection, collection_name=cfg.collection_name)
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
            raise ValueError(f"Unknown prefetch mode: {m!r} (expected 'hybrid' or 'sparseonly')")
    return result


def parse_rescorer_list(s: str) -> list[str]:
    rescorers = [x.strip() for x in s.split(",") if x.strip()]
    for r in rescorers:
        if r not in ("colbert", "cross-encoder", "rrf"):
            raise ValueError(f"Unknown rescorer: {r!r} (expected colbert, cross-encoder or rrf)")
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
        default=500,
        help="number of real corpus queries to sample for evaluation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", default="3", help="comma-separated list of k values, e.g. 10,20")
    parser.add_argument(
        "--prefetch-limit", default="5", help="comma-separated list of prefetch_limit values"
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
        "cross-encoder (local fastembed rerank), rrf (no second stage; needs hybrid mode)",
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
        )
    )


if __name__ == "__main__":
    main()
