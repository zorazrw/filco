"""Utility Functions."""

import json


def load_dataset(path: str) -> list[dict]:
    """Load dataset from JSON or JSONL file."""
    if path.endswith(".json"):
        return json.load(open(path, "r"))
    elif path.endswith(".jsonl"):
        return [json.loads(line.strip()) for line in open(path, "r")]
    else:
        extension = path.split(".")[-1]
        raise ValueError(f"File extension [{extension}] not valid.")


def write_dataset(path: str, dataset: list[dict]):
    """Write dataset to JSON or JSONL file."""
    if path.endswith(".json"):
        json.dump(dataset, open(path, "w"))
    elif path.endswith(".jsonl"):
        with open(path, "w") as fw:
            for res_dict in dataset:
                fw.write(json.dumps(res_dict) + "\n")
    else:
        extension = path.split(".")[-1]
        raise ValueError(f"File extension [{extension}] not valid.")

