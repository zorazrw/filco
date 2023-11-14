"""Query Llama Model on Mctx and Mgen Data."""

import argparse
# from query import clean_output
from utils import load_dataset, write_dataset
from xturing.models.base import BaseModel
from xturing.models.llama2 import Llama2Lora


def clean_output(text: str) -> str:
    text = text.lower()
    if "predictions:" in text:
        text = text.split("predictions:")[-1].strip()
    return text



def main():
    dataset = load_dataset(args.dataset_path)
    if args.max_num_examples is not None:
        dataset = dataset[: args.max_num_examples]
    print(f"Load #{len(dataset)} Examples. Start Inference ...")
    # model = BaseModel.create(args.model_name)
    # if args.model_path is not None:
    #     model = model.load_from_local(args.model_path)
    assert args.model_name == "llama2_lora"
    model = Llama2Lora(args.model_path)

    input_list = [ex["instruction"]+'\n\n'+ex["text"] for ex in dataset]
    predictions = model.generate(texts=input_list, batch_size=args.generate_batch_size)
    print("Finished Inference!")

    prediction_list = [[p] for p in predictions]
    cleaned_prediction_list = [[clean_output(p)] for p in predictions]

    output_list = [ex["target"] for ex in dataset]
    assert len(prediction_list) == len(cleaned_prediction_list) == len(input_list) == len(output_list)
    
    result_list = []
    for p,c,i,o in zip(prediction_list, cleaned_prediction_list, input_list, output_list):
        res_dict = {
            "input": i, "output": o,
            "predictions": p, "pred_answers": c,
        }
        result_list.append(res_dict)
    
    write_dataset(path=args.output_path, dataset=result_list)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path/Directory to the dataset (xturing processed version).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama2_lora",
        help="Name of the model to use (supported by xturing).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the prediction file.",
    )
    parser.add_argument(
        "--generate_batch_size",
        type=int,
        default=8,
        help="Batch size for generation.",
    )

    parser.add_argument(
        "--max_num_examples",
        type=int,
        default=None
    )

    args = parser.parse_args()

    main()
