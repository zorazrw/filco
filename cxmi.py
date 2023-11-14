"""Conditional Cross-Mutual Information (CXMI) Score."""

import math

import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Input Preparation
def get_input_text(
    question: str,
    question_prefix: str = "question",
    context: str = None,
    context_prefix: str = "context",
) -> str:
    """Construct the input text."""
    q_text = f"{question_prefix}: {question}"
    if context is None:
        return q_text
    ctx_text = f"{context_prefix}: {context}"
    return "\n".join([ctx_text, q_text])


def get_example_inputs(
    question: str,
    context: str,
    answers: list[str],
    question_prefix: str = "question",
    context_prefix: str = "context",
) -> dict:
    """Get example inputs for the generation model."""
    base_input = get_input_text(
        question,
        context=None,
        question_prefix=question_prefix,
        context_prefix=context_prefix,
    )
    ctx_input = get_input_text(
        question,
        context=context,
        question_prefix=question_prefix,
        context_prefix=context_prefix,
    )
    return {
        "base_input": base_input,
        "ctx_input": ctx_input,
        "answers": answers,
    }


# Score Calculation
def get_output_probs(
    model, tokenizer, input_text: str, output_text: str
) -> list[float]:
    """Compute the output probabilities of the output text given the input text."""
    input_dict = tokenizer(input_text, return_tensors="pt")  # <1, in-len>
    output_dict = tokenizer(output_text, return_tensors="pt")  # <1, out-len>
    input_dict["labels"] = output_dict["input_ids"]
    input_dict = {k: v.to(model.device) for k, v in input_dict.items()}

    logits = model(**input_dict).logits  # <1, out-len, vocab-size>
    probs = logits.softmax(dim=-1).squeeze(0)  # <out-len, vocab-size>
    label_ids = input_dict["labels"].squeeze(0).unsqueeze(-1)  # <out-len, 1>
    probs = probs.gather(1, label_ids).squeeze(-1)  # <out-len>
    return probs.tolist()


def sent_wise_diff(base_scores: list[float], ctx_scores: list[float]) -> float:
    """Compute the sentence-wise difference between context over the raw vector."""
    assert len(base_scores) == len(
        ctx_scores
    ), "The two lists must have the same length."
    # print("Base Scores: ", base_scores)
    # print("Ctx  Scores: ", ctx_scores)
    return -np.log(np.prod(base_scores) / np.prod(ctx_scores))


def sigmoid(x: float) -> float:
    """Sigmoid function."""
    return 1 / (1 + math.exp(-x))


def calc_cxmi_score(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    answer: str,
    base_input: str,
    ctx_input: str,
    apply_sigmoid: bool = False,
) -> float:
    """Compute the CXMI score."""
    base_probs = get_output_probs(model, tokenizer, base_input, answer)
    ctx_probs = get_output_probs(model, tokenizer, ctx_input, answer)
    diff = sent_wise_diff(base_scores=base_probs, ctx_scores=ctx_probs)
    if apply_sigmoid:
        diff = sigmoid(diff)
    return diff

