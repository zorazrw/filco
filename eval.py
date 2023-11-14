"""Evaluate Functions."""

import argparse
import re
import string
from collections import Counter

from utils import load_dataset


def clean_output(text: str) -> str:
    """Extract out the answer text portion.

    prediction: e.g., "The above context is (-23.0) helpful. answer: Violet Alva"
    returns: e.g., "Violent Alva"
    """
    text = text.replace("Answer", "answer")
    if "answer:" in text:
        text = text.split("answer:")[-1].strip()

    text = text.replace("Response", "response")
    if "response:" in text:
        text = text.split("response:")[-1].strip()

    text = text.replace("Judgement", "judgement")
    if "judgement:" in text:
        text = text.split("judgement:")[-1].strip()

    text = text.replace("Score", "score")
    if "score:" in text:
        text = text.split("score:")[-1].strip()
    if "answer:" in text:
        text = text.split("answer:")[0].strip()

    return text



def normalize_text(text: str) -> str:
    """Normalize text with lowercasing, removing articles, and punctuation."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(lower(text)))


def has_answer(text: str, answers: list[str]) -> float:
    """Check if text contains any of the answers."""
    return float(any([(ans.lower() in text.lower()) for ans in answers]))


def calc_exact_match(text: str, answers: list[str]) -> bool:
    """Check if prediction is exactly the same as any of the answers."""
    norm_text = normalize_text(text)
    norm_answers = [normalize_text(ans) for ans in answers]
    return max([(norm_text == norm_ans) for norm_ans in norm_answers])


def calc_unigram_f1(text: str, answers: list[str], field: str = "f1") -> float:
    """Calculate unigram f1 score between the text and reference answers."""
    norm_pred = normalize_text(text)
    norm_answers = [normalize_text(ans) for ans in answers]
    common_tokens = [
        Counter(norm_pred) & Counter(norm_ans) for norm_ans in norm_answers
    ]
    num_same = [sum(common.values()) for common in common_tokens]

    score_list = []
    for i, num in enumerate(num_same):
        if num == 0:
            score_list.append(0.0)
        else:
            p = 1.0 * num / len(norm_pred)
            r = 1.0 * num / len(norm_answers[i])
            f1 = 2 * p * r / (p + r)
            if field == "precision":
                score_list.append(p)
            elif field == "recall":
                score_list.append(r)
            elif field == "f1":
                score_list.append(f1)
            else:
                raise ValueError(f"Unknown field: {field}")
    return max(score_list)


EvalDict = {"em": calc_exact_match, "f1": calc_unigram_f1}


def main():
    """Main evaluation function."""
    dataset = load_dataset(args.dataset_path)  # list[dict]
    predset = load_dataset(args.predset_path)  # list[dict]
    assert len(dataset) == len(predset)

    eval_func = EvalDict[args.metric_name]
    scores = [
        eval_func(clean_output(pex[args.predict_key][0]), dex["answers"])
        for pex, dex in zip(predset, dataset)
    ]
    print(
        f"Average {args.metric_name.upper()} score:", f"{sum(scores) / len(scores):.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data i/o
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--predset_path", type=str, required=True)
    parser.add_argument("--predict_key", type=str, default="pred_answers")

    parser.add_argument("--metric_name", type=str, default="em", choices=["em", "f1"])

    args = parser.parse_args()

    main()

