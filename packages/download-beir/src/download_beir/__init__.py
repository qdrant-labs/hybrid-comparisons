import json
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset


def download_dataset(name: str, take: int = 100_000) -> list[dict[str, Any]]:
    d = load_dataset(f"Beir/{name}", split="train", streaming=True)
    dataset = d.take(take)
    return list(dataset)


def format_dataset(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    new_data = []
    for i, d in enumerate(data):
        new_data.append({"pid": i, "text": d["text"], "queries": [d["query"]]})
    return new_data


def write_dataset(data: list[dict[str, Any]], path: str) -> None:
    if not (par := Path(path).parent).exists():
        par.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as f:
        f.writelines([json.dumps(d) + "\n" for d in data])


def main() -> None:
    args = sys.argv
    if len(args) < 2:
        print("You need to provide the dataset name")
        sys.exit(1)
    name = args[1]
    take = 100_000
    if len(args) > 2:
        take = int(args[2])
    data = download_dataset(name, take)
    new_data = format_dataset(data)
    write_dataset(
        new_data, f"../../data/{name.replace('-generated-queries', '')}/corpus.jsonl"
    )
