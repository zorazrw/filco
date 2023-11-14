"""Convert Mctx and Mgen Data to xTuring Style."""

import argparse
from utils import load_dataset, write_dataset


MCTX_INSTR_MAP = {
    "nq": "Given the ['question', 'context'], predict the most helpful sentences in the context.",
    "tqa": "Given the ['question', 'context'], predict the most helpful sentences in the context.",
    "fever": "Given the ['claim', 'context'], predict the most helpful sentences in the context.",
    "wow": "Given the ['query', 'context'], predict the most helpful sentences in the context.",
    "hotpotqa": "Given the ['question', 'context'], predict the most helpful sentences in the context.",
    "eli5": "Given the ['question', 'context'], predict the most helpful sentences in the context.",
}

MGEN_INSTR_MAP = {
    "nq": "Given the ['context', 'question'], predict the answer to the question.",
    "tqa": "Given the ['context', 'question'], predict the answer to the question.",
    "fever": "Given the ['context', 'claim'], predict the judgement to the claim.",
    "wow": "Given the ['context', 'query'], predict the response to the query.",
    "hotpotqa": "Given the ['context', 'question'], predict the answer to the question.",
    "eli5": "Given the ['context', 'question'], predict the answer to the question.",
}

KEYWORD_MAP = {
    "nq": "question:",
    "tqa": "question:",
    "fever": "claim:",
    "wow": "query:",
    "hotpotqa": "question:",
    "eli5": "question:",
}


def main():
    src_data = load_dataset(args.input_data_path)

    if (args.dataset_type == "mctx"):
        instruct_text = MCTX_INSTR_MAP[args.dataset_name]
        keyword = KEYWORD_MAP[args.dataset_name]
    else: 
        instruct_text = MGEN_INSTR_MAP[args.dataset_name]
        keyword = "context:"

    tgt_data = []
    for i,ex in enumerate(src_data):
        if keyword not in ex["input"]: print(i, " | ", ex["input"])
        tgt_data.append({
            "instruction": instruct_text,
            "text": ex["input"][ex["input"].index(keyword): ],
            "target": ex["output"],
        })
    assert args.output_path.endswith(".jsonl")
    write_dataset(args.output_path, tgt_data)

    if args.print_example:
        for k,v in tgt_data[0].items():
            print(f"=== {k.upper()} ===")
            print(v)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data_path", type=str, required=True,
        help="Path to the original data.",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Directory to the output data.",
    )

    parser.add_argument(
        "--dataset_name", type=str, required=True,
        choices=["nq", "tqa", "fever", "wow", "hotpotqa", "eli5"],
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--dataset_type", type=str, required=True,
        choices=["mctx", "mgen"],
        help="Type of the dataset.",
    )
    parser.add_argument(
        "--print_example", action="store_true",
    ) 
    

    args = parser.parse_args()

    main()
