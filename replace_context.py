"""Replace the 'context' in the input with 'sent' in the output. """

import argparse
from typing import Dict, List
from get_inputs import get_answer, prefix_format, CONTEXT_PREFIX
from utils import load_dataset, write_dataset

def main():
    dataset = load_dataset(args.dataset_path)
    predset = load_dataset(args.predset_path)
    # assert len(dataset) == len(predset)
    N = len(predset)

    def get_input_text(input_text: str, output_text: str) -> str: 
        prefix, text = input_text.split("\n\n")
        prefix = prefix.replace("['question', 'context']", "['context', 'question']")
        prefix = prefix.replace("most helpful sentence", "answer to the question")
        question, context = text.split("\ncontext:")
        question = question.rstrip()
        sent_ctx = prefix_format.format(CONTEXT_PREFIX, output_text)
        return prefix + '\n\n' + sent_ctx + '\n' + question

    def get_input_text_wow(input_text: str, output_text: str) -> str:
        prefix, text = input_text.split("\n\n")
        prefix = prefix.replace("question", "query")
        prefix = prefix.replace("['query', 'context']", "['context', 'query']")
        prefix = prefix.replace("most helpful sentence", "response to the conversation")
        question, context = text.split("context:")
        question = question.rstrip()
        context = "context:" + context
        sent_ctx = prefix_format.format(CONTEXT_PREFIX, output_text)
        return '\n\n'.join([prefix, sent_ctx, question])

    def get_input_text_fever(input_text: str, output_text: str) -> str:
        prefix, text = input_text.split("\n\n")
        prefix = prefix.replace("['claim', 'context']", "['context', 'claim']")
        prefix = prefix.replace("most helpful sentence", "judgement to the claim ('SUPPORTS' or 'REFUTES')")
        prefix = prefix.replace("answer to the question", "judgement to the claim ('SUPPORTS' or 'REFUTES')")
        question, context = text.split('context:')
        question = question.rstrip()
        sent_ctx = prefix_format.format(CONTEXT_PREFIX, output_text)
        return prefix + '\n\n' + sent_ctx + '\n' + question

    out_set = []
    for i, (dex, pex) in enumerate(zip(dataset, predset)):
        if args.process_dataset == "wow": 
            input_text = get_input_text_wow(pex["input"], pex["output"])
        elif args.process_dataset == "fever": 
            input_text = get_input_text_fever(pex["input"], pex["output"])
        else:
            input_text = get_input_text(pex["input"], pex["output"])
        output_text = get_answer(dex, answer_prefix=args.answer_prefix)
        out_set.append({
            "input": input_text,
            "output": output_text,
        })
    write_dataset(path=args.output_path, dataset=out_set)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/base/NQ/test.json", 
                        help="Source to get answer of questions.")
    parser.add_argument("--predset_path", type=str, required=True,
                        help="Get model original inputs and outputs.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save aggregated dataset.")
    parser.add_argument("--process_dataset", type=str, default="nq", choices=["nq", "tqa", "wow", "fever", "hotpotqa", "eli5"])
    parser.add_argument("--answer_prefix", type=str, default="answer", choices=["answer", "response", "judgement"])
    args = parser.parse_args()

    main()
