
# FilCo

<p align="left">
  <a href="http://creativecommons.org/licenses/by-sa/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--SA%204.0-green.svg"></a>
  <a href="https://arxiv.org/abs/2311.08377"><img src="https://img.shields.io/badge/arXiv-2311.08377-b31b1b.svg"></a>
</p>

This repository contains the code and data about the project:
[Learning to Filter Context for Retrieval-Augmented Generation](https://arxiv.org/pdf/2311.08377.pdf)

## Install

Install all required libraries by running

```bash
pip install -r requirements.txt
```

Retrieve top relevant Wikipedia passages using [Dense Passage Retriever (DPR)](https://github.com/facebookresearch/DPR)
and store into the `./datasets/${name}` directory. We also provide preprocessed datasets with top-5 retrieved passages [(here)](https://drive.google.com/file/d/13z_qrVOBlgu75IJBpX-1vMSCC6hC9yH4/view?usp=sharing).
We specify `${name}` for six datasets with ['nq', 'tqa', 'hotpotqa', 'fever', 'wow'] in following example commands.

## Measure Retrieved Passages

Before filtering out potentially redundant context, we need to measure the utility scores of individual spans in the retrieved passages. 
You can use any of the three context filtering strategies:
(i) entailment, (ii) lexical overlap, and (iii) conditional cross-mutual information (CXMI).

Use `measure_ctxs.py` to measure the utility score of each retrieved passage,
as well as individual sentences within, for example:

```bash
python measure_ctxs.py \
--dataset_path "./datasets/nq/base/test.json" \
--output_path  "./datasets/nq/scored/test.json" \
--metric_name  "strinc" "lexical" "cxmi" \
--n_contexts 5 \
--prefix "Given the ['context', 'question'], predict the answer to the question:"
```

If "cxmi" is specified as one of the `metric_name`s, make sure you specify the huggingface model to use in `model_name_or_path`. Or it will use "google/flan-t5-xl" by default.

## Obtain Training & Testing Data

Use `get_inputs.py` to create input-output training pairs for both the context filtering model $M_{ctx}$ and generation model $M_{gen}$.

For the _context filtering task_, the input should be all top-K retrieved passages, and the output is context filtered with one of the three strategies.

```bash
python get_inputs.py \
--dataset_path "./datasets/nq/scored/train.json" \
--output_path "./datasets/nq/mctx/em/train_em_top1.json" \
--input_list question passage --output_list filtered \
--n_examples 0 --n_contexts 1 \
--filter_criteria strinc --print_example
```

Alter the value of `n_examples` to include more in-context examples. Adjust the value of `n_contexts` to change the number of retrieved passages involved. `filter_criteria` specifies which filtering strategy you want to use, among ['strinc', 'lexical', 'cxmi'].

For the _generation task_, the input should be filtered context, and output is the annotated output.

```bash
python get_inputs.py \
--dataset_path "./datasets/nq/scored/train.json" \
--output_path "./datasets/nq/mgen/em/train_em_top1.json" \
--input_list question filtered --output_list answer \
--n_examples 0 --n_contexts 1 \
--filter_criteria strinc --print_example
```

The only changes to the context filtering case is the `input_list` and `output_list`, where we switched the input context to from entire passages ('passage') to filtered sentences ('filtered').

## Training A Context Filtering Model

Perform the above processing on training, validation, and test data,
then to fine-tune a FlanT5 (xl) model using `train.py`, which passes
in "google/flan-t5-xl" to the `model_name_or_path` argument by default.

```bash
python train.py \
--train_data_path "./datasets/nq/mctx/em/train_em_top1.json" \
--eval_data_path "./datasets/nq/mctx/em/dev_em_top1.json" \
--test_data_path "./datasets/nq/mctx/em/test_em_top1.json" \
--output_dir "./checkpoints/nq-mctx_filco-em" \
--do_train --do_eval --do_predict
```

After training, load the fine-tuned checkpoint to predict filtered context for testing examples.

```bash
python query.py \
--dataset_path "./datasets/nq/mctx/em/test_em_top1.json" \
--output_path "./output/nq/mctx/filco-em_tuned-ft5.json" \
--model_name_or_path "./checkpoints/nq-mctx_filco-em"
```

After this, convert the dataset to generation example format by

```bash
python replace_context.py \
--dataset_path "./datasets/nq/base/test.json" \
--predset_path "./output/nq/mctx/filco-em_tuned-ft5.json" \
--output_path "./datasets/nq/mgen/em/test_em_top1_predict-ft5.json" \
--process_dataset nq
```

To train and query LLaMa models, switch the model name to "meta-llama/Llama-2-7b-hf".
Alternatively using xTuring, run `train_llama.py` and `query_llama.py` with similar arguments, but transform the examples into instruction style using `convert_dataset.py`.

## Training A Generation Model with Filtered Context

Prepare the training and validation data using the same method,
then train Flan-T5 models using `train.py` and LLaMa models with `train_llama.py`.

```bash
python train.py \
--train_data_path "./datasets/nq/mgen/em/train_em_top1.json" \
--eval_data_path "./datasets/nq/mgen/em/dev_em_top1.json" \
--test_data_path "./datasets/nq/mgen/em/test_em_top1.json" \
--output_dir "./checkpoints/nq-mgen_filco-em" \
--do_train --do_eval --do_predict
```

To use the tuned model checkpoint for inference, run

```bash
python query.py \
--dataset_path "./datasets/nq/mgen/em/test_em_top1.json" \
--output_path "./output/nq/mgen/silver-em_tuned-ft5.json" \
--model_name_or_path "./checkpoints/nq-mgen_filco-em"
```

Switch the silver filtered context (e.g., "./datasets/nq/mgen/em/train_em_top1.json") to model filtered context (e.g., "./output/nq/mctx/filco-em_tuned-ft5.json") to experiment in the __FilCo__ setting.

## Evaluating Filtering and Generation Models

To evaluate the generation performance, use the EM (~Accuracy) or F1
according to the task formulation.

```bash
python eval.py \
--dataset_path "./datasets/nq/base/test.json" \
--predset_path "./output/nq/mgen/silver-em_tuned-ft5.json" \
--metric_name "em"
```

## Reference

If you find our paper or code useful, please cite the paper

```
@article{wang2023learning,
  title={Learning to Filter Context for Retrieval-Augmented Generation},
  author={Zhiruo Wang, Jun Araki, Zhengbao Jiang, Md Rizwan Parvez, Graham Neubig},
  journal={arXiv preprint arXiv:2311.08377},
  year={2023}
}
```
