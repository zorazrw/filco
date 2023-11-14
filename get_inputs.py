"""Create I/O to Evaluate/Train Models.

Default I/O for Context Filtering: [i] question context [o] sent
Default I/O for Output Generation: [i] sent question [o] answer
"""

import argparse

from eval import has_answer
from utils import load_dataset, write_dataset

# Individual Components
QUESTION_PREFIX = "question"
ANSWER_PREFIX = "answer"
CONTEXT_PREFIX = "context"
prefix_format = "{}: {}"


def get_question(
    example: dict,
    question_prefix: str = QUESTION_PREFIX,
    add_prefix: bool = True,
) -> str:
    """Get the question from the example."""
    question = example["question"]
    if add_prefix:
        question = prefix_format.format(question_prefix, question)
    return question


def get_context(
    example: dict,
    n_contexts: int = 1,
    context_prefix: str = CONTEXT_PREFIX,
    add_prefix: bool = True,
) -> str:
    """Get the context from the example."""
    context_list = [ctx["text"] for ctx in example["ctxs"][:n_contexts]]
    context = '\n'.join(context_list)
    if add_prefix:
        context = prefix_format.format(context_prefix, context)
    return context


def get_sent(
    example: dict, 
    n_contexts: int = 1, 
    criteria: str = "strinc",
    num_sents: int = None, 
    threshold: float = None, 
) -> str:
    """Get the best sentence from contexts."""
    sentences = []
    if threshold is None:
        threshold = 0.0
    for idx in range(n_contexts):
        if criteria == "strinc":
            for sent_dict in example["ctxs"][idx]["sentences"]:
                if sent_dict[criteria] is True:
                    sentences.append(sent_dict["text"])
                    # break
        else:
            if num_sents is None:
                num_sents = len(example["ctxs"][idx]["sentences"])
            ctx_sents = sorted(
                example["ctxs"][idx]["sentences"], 
                key=lambda x: -x[criteria]
            )
            sentences.extend([
                s["text"] for s in ctx_sents[: num_sents] 
                if s[criteria] >= threshold
            ])
    sent_text = " ".join(sentences)
    return sent_text


def get_answer(
    example: dict,
    answer_prefix: str = ANSWER_PREFIX,
    find_best: bool = True,
    n_contexts: int = 1,
    add_prefix: bool = True,
) -> str:
    """Find the answer index that best possibly in the context.

    Using the top-1 retrieved context by default.
    """
    if find_best:
        for idx in range(n_contexts):
            context = example["ctxs"][idx]["text"].lower()
            answer_exists = [
                has_answer(context, [ans.lower()]) for ans in example["answers"]
            ]
            if any(answer_exists):
                answer_text = example["answers"][answer_exists.index(True)]
                break
        else:
            answer_text = example["answers"][0]
    else:
        answer_text = example["answers"][0]

    if add_prefix:
        answer_text = prefix_format.format(answer_prefix, answer_text)
    return answer_text


# Example Creation Functions
def get_example_io(
    example: dict,
    input_list: list[str],
    output_list: list[str],
    n_contexts: int = 1,
    num_sents: int = None, 
    threshold: float = None,
    filter_criteria: str = "strinc",
    question_prefix: str = "question",
    answer_prefix: str = "answer",
    context_prefix: str = "context",
) -> tuple[str, str]:
    """Get input and output texts for the given example."""
    input_text_list, output_text_list = [], []
    for inp in input_list:
        if inp == "question":
            input_text_list.append(
                get_question(example, question_prefix=question_prefix)
            )
        elif inp == "passage":
            input_text_list.append(get_context(example, n_contexts, context_prefix=context_prefix))
        elif inp == "filtered":
            sent = get_sent(
                example=example, 
                n_contexts=n_contexts, 
                criteria=filter_criteria,
                num_sents=num_sents,
                threshold=threshold, 
            )
            if not sent.strip():
                sent = get_context(example, context_prefix=context_prefix)
            else:
                sent = prefix_format.format(CONTEXT_PREFIX, sent)
            input_text_list.append(sent)
        else:
            raise ValueError(f"Invalid input type {inp}")
    input_text = "\n".join(input_text_list)

    for out in output_list:
        if out == "answer":
            output_text_list.append(
                get_answer(
                    example,
                    answer_prefix=answer_prefix,
                    n_contexts=n_contexts,
                )
            )
        elif out == "filtered":
            output_text_list.append(
                get_sent(
                    example=example,
                    n_contexts=n_contexts,
                    criteria=filter_criteria,
                    num_sents=num_sents,
                    threshold=threshold,
                )
            )
        else:
            raise ValueError(f"Invalid output type {out}")
    output_text = "\n".join(output_text_list)

    return input_text, output_text


# ICT Example Creation Functions
def get_ict_io(
    example: dict,
    in_context_examples: list[dict],
    input_list: list[str],
    output_list: list[str],
    no_prefix: bool = False,
    filter_criteria: str = "strinc",
    n_contexts: int = 1,
    num_sents: int = None,
    threshold: float = None,
    question_prefix: str = "question",
    answer_prefix: str = "answer",
    context_prefix: str = "context",
) -> tuple[str, str]:
    """Get input and output texts with in-context examples."""
    ict_io_list = []
    for example in in_context_examples:
        itext, otext = get_example_io(
            example,
            input_list,
            output_list,
            n_contexts=n_contexts,
            num_sents=num_sents,
            threshold=threshold,
            filter_criteria=filter_criteria,
            question_prefix=question_prefix,
            answer_prefix=answer_prefix,
            context_prefix=context_prefix,
        )
        ict_io_list.append("\n".join([itext, otext]))
    input_text, output_text = get_example_io(
        example,
        input_list,
        output_list,
        n_contexts=n_contexts,
        num_sents=num_sents,
        threshold=threshold,
        filter_criteria=filter_criteria,
        question_prefix=question_prefix,
        answer_prefix=answer_prefix,
        context_prefix=context_prefix,
    )

    if no_prefix:
        prefix = ""
    else:
        input_text_list = []
        for ii in input_list:
            if (ii == "filtered") or (ii == "passage"):
                input_text_list.append(context_prefix)
            elif ii == "question":
                input_text_list.append(question_prefix)
            else:
                input_text_list.append(ii)

        output_text_list = []
        for oo in output_list:
            if oo == "filtered":
                output_text_list.append(
                    f"most helpful sentence in the {context_prefix}"
                )
            elif oo == "answer":
                if answer_prefix == "response":
                    output_text_list.append("response to the query")
                elif answer_prefix == "judgement":
                    output_text_list.append("judgement to the claim")
                else:
                    output_text_list.append("answer to the question")

        if len(output_text_list) == 1:
            prefix = f"Given the {input_text_list}, predict the {output_text_list[0]}."
        else:
            prefix = (
                f"Given the {input_text_list}, "
                f"predict the {output_text_list[0]} first, "
                f"then predict the {output_text_list[1]}."
            )

        if question_prefix == "claim" and answer_prefix == "judgement":
            prefix += (
                "('SUPPORTS' or 'REFUTES')\n"
                "If the 'context' does not provide enough information "
                "to judge the claim, use your own knowledge instead."
            )

    full_input_text = "\n\n".join([prefix] + ict_io_list + [input_text])
    return full_input_text.strip(), output_text.strip()


def main():
    """Run the main data processing function."""
    dataset = load_dataset(args.dataset_path)
    N = len(dataset)

    def get_examples(index: int, n_examples: int) -> list[int]:
        """Get indices of in-context examples."""
        indices = [(index - i - 1) % N for i in range(n_examples)]
        return [dataset[i] for i in indices]

    procset = []
    for idx, ex in enumerate(dataset):
        input_text, output_text = get_ict_io(
            example=ex,
            in_context_examples=get_examples(idx, args.n_examples),
            input_list=args.input_list,
            output_list=args.output_list,
            no_prefix=args.no_prefix,
            filter_criteria=args.filter_criteria,
            n_contexts=args.n_contexts,
            num_sents=args.num_sents,
            threshold=args.threshold,
            question_prefix=args.question_prefix,
            answer_prefix=args.answer_prefix,
            context_prefix=args.context_prefix,
        )
        procset.append({"input": input_text, "output": output_text})

    write_dataset(args.output_path, procset)

    if args.print_example:
        example = procset[0]
        for k, v in example.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument(
        "--input_list",
        type=str,
        nargs="+",
        default=["passage", "question"],
        choices=["passage", "filtered", "question"],
        help="List of keys to use as input.",
    )
    parser.add_argument(
        "--output_list",
        type=str,
        nargs="+",
        default=["answer"],
        choices=["filtered", "answer"],
    )
    parser.add_argument("--no_prefix", action="store_true")
    parser.add_argument("--n_examples", type=int, default=0)

    # sent config
    parser.add_argument("--n_contexts", type=int, default=1)
    parser.add_argument(
        "--filter_criteria", type=str, default="strinc", 
        choices=["strinc", "lexical", "cxmi"]
    )
    parser.add_argument("--num_sents", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=None)

    # prefix config
    parser.add_argument("--question_prefix", type=str, default="question")
    parser.add_argument("--answer_prefix", type=str, default="answer")
    parser.add_argument("--context_prefix", type=str, default="context")

    parser.add_argument("--print_example", action="store_true")

    args = parser.parse_args()

    if len(args.output_list) == 0:
        raise ValueError("Must have at least one output type (`answer` or `sent`).")
    
    if "filtered" in args.output_list:
        assert (args.num_sents is not None) or (args.threshold is not None), \
        f"Must specify either `num_sents` or `threshold` for `filtered` output."

    main()

