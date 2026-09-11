import json

from pathlib import Path
from typing import Any
from datasets import load_dataset

DATASET = "BeIR/dbpedia-entity-generated-queries"
DATASET_PATH = "../../data/dbpedia/corpus.jsonl"

def download_dataset() -> list[dict[str, Any]]:
    d = load_dataset(DATASET, split="train", streaming=True)
    dataset = d.take(100_000)
    return list(dataset)

def format_dataset(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    new_data = []
    for (i, d) in enumerate(data):
        new_data.append({"pid": i, "text": d["text"], "queries": [d["query"]]})
    return new_data

def write_dataset(data: list[dict[str, Any]], path: str) -> None:
    if not (par := Path(path).parent).exists():
        par.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as f:
        f.writelines([json.dumps(d) + "\n" for d in data])

def main() -> None:
    data = download_dataset()
    new_data = format_dataset(data)
    write_dataset(new_data, DATASET_PATH)
