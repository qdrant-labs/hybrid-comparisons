# pyright: basic
"""Downloads the first 100k rows (text + associated real search queries) of
jordane95/msmarco-passage-corpus-with-query and writes them out as JSONL:
one {"pid": i, "text": ..., "queries": [...]} object per line.
"""

import gzip
import json
import sys

from huggingface_hub import hf_hub_download

REPO_ID = "jordane95/msmarco-passage-corpus-with-query"
N_ROWS = 100_000


def main() -> None:
    path = sys.argv[1] if len(sys.argv) >= 2 else "corpus.jsonl"
    src_path = hf_hub_download(
        repo_id=REPO_ID, filename="corpus.jsonl.gz", repo_type="dataset"
    )

    with gzip.open(src_path, "rt") as src, open(path, "w") as out:
        for pid, line in enumerate(src):
            if pid >= N_ROWS:
                break
            row = json.loads(line)
            text = row["text"].replace("\n", " ").replace("\t", " ")
            queries = [q.strip() for q in row["queries"] if q and q.strip()]
            out.write(json.dumps({"pid": pid, "text": text, "queries": queries}) + "\n")
    print(f"Successfully wrote {N_ROWS} passages (with queries) to {path}")

if __name__ == "__main__":
    main()
