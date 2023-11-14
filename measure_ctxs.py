"""Calculate Scores of Individual Sentences in Retrieved Passages."""

import argparse

import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from cxmi import calc_cxmi_score, get_example_inputs
from eval import calc_unigram_f1, has_answer
from utils import load_dataset, write_dataset


def calc_cxmi(
    text: str,
    question: str,
    answers: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
) -> float:
    """Calculate CXMI score for a context text."""
    proc_inputs = get_example_inputs(
        question=args.prefix + question,
        context=text,
        answers=answers,
    )
    cxmi_score = calc_cxmi_score(
        model=model,
        tokenizer=tokenizer,
        answer=proc_inputs["answers"][0],
        base_input=proc_inputs["base_input"],
        ctx_input=proc_inputs["ctx_input"],
        apply_sigmoid=True,
    )
    return cxmi_score


def main():
    """Run the main context measuring function."""
    # load dataset
    dataset = load_dataset(args.dataset_path)

    if "cxmi" in args.metric_names:
        if args.tokenizer_name_or_path is None:
            args.tokenizer_name_or_path = args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, cache_dir="/scratch/zhiruow", torch_dtype=torch.bfloat16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
    else:
        tokenizer, model = None, None

    def calc_text_scores(
        text: str, question: str, answers: list[str], metric_names: list[str]
    ) -> dict[str, float]:
        """Calculate scores for a context text."""
        scores_dict = {}
        if "strinc" in metric_names:
            scores_dict["strinc"] = has_answer(text, answers)
        if "lexical" in metric_names:
            scores_dict["lexical"] = calc_unigram_f1(text, answers)
        if "precision" in metric_names:
            scores_dict["precision"] = calc_unigram_f1(text, answers, field="precision")
        if "recall" in metric_names:
            scores_dict["recall"] = calc_unigram_f1(text, answers, field="recall")
        if "cxmi" in metric_names:
            scores_dict["cxmi"] = calc_cxmi(text, question, answers, tokenizer, model)
        scores_dict["text"] = text
        return scores_dict

    def calc_sentence_scores(
        ctx_text: str,
        question: str,
        answers: list[str],
        metric_names: list[str],
    ) -> list[dict]:
        """Calculate scores for each sentence in a context text."""
        sentences = sent_tokenize(ctx_text)
        sent_dicts = [
            calc_text_scores(s, question, answers, metric_names) for s in sentences
        ]
        return sent_dicts

    sentset = []

    for i, ex in enumerate(dataset):
        ctxs = []
        for j, ctx in enumerate(ex["ctxs"][: args.n_contexts]):
            # passage-wise measure
            example = calc_text_scores(
                ctx["text"], ex["question"], ex["answers"], args.metric_names
            )
            example.update({"title": ctx["title"], "text": ctx["text"]})

            # sentence-wise measure
            sent_scores = calc_sentence_scores(
                ctx["text"], ex["question"], ex["answers"], args.metric_names
            )
            example["sentences"] = sent_scores
            ctxs.append(example)

        sent_ex = {"question": ex["question"], "answers": ex["answers"], "ctxs": ctxs}
        sentset.append(sent_ex)

        if (i + 1) % args.report_steps == 0:
            print(f"Processed {i+1} examples.")

    write_dataset(path=args.output_path, dataset=sentset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--metric_names",
        type=str,
        nargs="+",
        default=["strinc", "lexical", "cxmi"],
        choices=["strinc", "lexical", "cxmi", "precision", "recall"],
    )
    parser.add_argument("--n_contexts", type=int, default=10)
    # if using 'cxmi' metric
    parser.add_argument(
        "--prefix",
        type=str,
        default=(
            "Given the ['context', 'question'], " "predict the answer to the question:"
        ),
    )
    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-xl")
    parser.add_argument(
        "--tokenizer_name_or_path", type=str, default="google/flan-t5-xl"
    )
    # logging
    parser.add_argument("--report_steps", type=int, default=500)
    args = parser.parse_args()

    if "cxmi" in args.metric_names:
        assert (
            args.model_name_or_path is not None
        ), "Need to specify `model_name_or_path` if using 'cxmi' metric."

    main()

