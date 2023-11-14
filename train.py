"""Tuning Huggingface Models."""
import os
os.environ["WANDB_DISABLED"] = "true"

import argparse
import logging

import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

import datasets
import evaluate
from query import clean_output

logger = logging.getLogger(__name__)


def main():
    """Run the main training function."""
    # load config, tokenizer, and model
    config = AutoConfig.from_pretrained(
        args.config_name if (args.config_name is not None) else args.model_name,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if (args.tokenizer_name is not None) else args.model_name,
        cache_dir=args.cache_dir,
    )
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.distribute_model:
        model_kwargs["device_map"] = "balanced_low_0"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        **model_kwargs,
    )

    # load train, validation, and test data
    data_files = {}
    if args.do_train:
        data_files["train"] = args.train_data_path
        extension = args.train_data_path.split(".")[-1]
    if args.do_eval:
        data_files["validation"] = args.eval_data_path
        extension = args.eval_data_path.split(".")[-1]
    if args.do_predict:
        data_files["test"] = args.test_data_path
        extension = args.test_data_path.split(".")[-1]
    raw_datasets = datasets.load_dataset(
        extension,
        data_files=data_files,
        cache_dir=args.cache_dir,
    )

    # column names to remove after preprocessing
    train_column_names, eval_column_names, test_column_names = None, None, None
    if args.do_train:
        train_column_names = raw_datasets["train"].column_names
    if args.do_eval:
        eval_column_names = raw_datasets["validation"].column_names
    if args.do_predict:
        test_column_names = raw_datasets["test"].column_names

    # tokenization config
    padding = "max_length" if args.pad_to_max_length else False
    max_seq_length = min(args.max_seq_length, config.n_positions)
    max_answer_length = args.max_answer_length

    # training args
    train_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "evaluation_strategy": args.evaluation_strategy,
        "eval_steps": args.eval_steps,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
    }
    train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}
    training_args = Seq2SeqTrainingArguments(
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        predict_with_generate=True,
        report_to="none",
        **train_kwargs,
    )

    # preprocessing
    def preprocess_function(examples: datasets.Dataset) -> dict:
        inputs = examples["input"]
        targets = examples["output"]
        model_inputs = tokenizer(
            inputs,
            max_length=max_seq_length,
            padding=padding,
            truncation=True,
        )  # {"input_ids": ..., "attention_mask": ...}
        labels = tokenizer(
            text_target=targets,
            max_length=max_answer_length,
            padding=padding,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset, eval_dataset, predict_dataset = None, None, None
    if args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_examples = raw_datasets["train"]
        if args.max_train_examples is not None:
            max_train_examples = min(args.max_train_examples, len(train_examples))
            train_examples = train_examples.select(range(max_train_examples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_examples.map(
                preprocess_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=train_column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
    if args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if args.max_eval_examples is not None:
            max_eval_examples = min(args.max_eval_examples, len(eval_examples))
            eval_examples = eval_examples.select(range(max_eval_examples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_examples.map(
                preprocess_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=eval_column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
    if args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if args.max_predict_examples is not None:
            predict_examples = predict_examples.select(range(args.max_predict_examples))
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_examples.map(
                preprocess_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=test_column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # data collator
    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    # metric
    metric = evaluate.load("exact_match")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds = [clean_output(dp) for dp in decoded_preds]
        p.label_ids[p.label_ids < 0] = 0
        decoded_labels = tokenizer.batch_decode(p.label_ids, skip_special_tokens=True)

        decoded_preds = [clean_output(dp) for dp in decoded_preds]
        decoded_labels = [clean_output(dl) for dl in decoded_labels]
        return metric.compute(predictions=decoded_preds, references=decoded_labels)

    # train, evaluate, and test
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if training_args.predict_with_generate
        else None,
    )

    # Training
    if args.do_train:
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        max_train_examples = (
            args.max_train_examples
            if (args.max_train_examples is not None)
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_examples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else max_answer_length
    )
    num_beams = (
        args.num_beams
        if args.num_beams is not None
        else training_args.generation_num_beams
    )

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=max_length, num_beams=num_beams, metric_key_prefix="eval"
        )

        max_eval_examples = (
            args.max_eval_examples
            if args.max_eval_examples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_examples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics

        max_predict_examples = (
            args.max_predict_examples
            if args.max_predict_examples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_examples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # hf name
    parser.add_argument("--model_name", type=str, default="google/flan-t5-xl")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--overwrite_cache", action="store_true")

    # data path
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--test_data_path", type=str, default=None)

    # preprocess config
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length (by tokens) for inputs.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=16,
        help="Maximum sequence length (by tokens) for answers.",
    )
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)

    # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
    # training config
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--distribute_model", action="store_true")
    # evaluation & save model
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps", "no"],
    )
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument(
        "--save_strategy", type=str, default="epoch", choices=["epoch", "steps", "no"]
    )
    parser.add_argument("--save_steps", type=int, default=None)
    # generation config
    parser.add_argument("--num_beams", type=int, default=1)
    # train/eval/test mode
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_eval_examples", type=int, default=None)
    parser.add_argument("--max_predict_examples", type=int, default=None)

    args = parser.parse_args()

    main()

