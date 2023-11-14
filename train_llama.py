"""Training LLaMa Model on Mctx and Mgen Tasks."""

import argparse
from xturing.models.base import BaseModel
from xturing.datasets.instruction_dataset import InstructionDataset


def main():
    dataset = InstructionDataset(args.train_data_path)
    if args.max_num_examples is not None:
        dataset.data["train"] = dataset.data["train"].select([i for i in range(args.max_num_examples)])
    model = BaseModel.create(args.model_name)

    model.finetuning_args.learning_rate = args.learning_rate
    model.finetuning_args.gradient_accumulation_steps = args.gradient_accumulation_steps
    model.finetuning_args.batch_size = args.batch_size
    model.finetuning_args.eval_steps = args.eval_steps
    model.finetuning_args.save_steps = args.save_steps
    model.finetuning_args.max_length = args.max_length
    model.finetuning_args.num_train_epochs = args.num_train_epochs
    model.finetuning_args.optimizer_name = args.optimizer_name
    model.finetuning_args.output_dir = args.output_dir
    
    print("Model Finetuning Arg:")
    print(model.finetuning_args)

    model.finetune(dataset=dataset)
    model.save(args.output_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path", type=str, required=True,
        help="Path/Directory to the training dataset (xturing processed version).",
    )
    parser.add_argument(
        "--model_name", type=str, default="llama2_lora",
        help="Name of the model to use (supported by xturing).",
    )

    parser.add_argument(
        "--max_num_examples", type=int, default=None,
        help="Maximum number of examples to use for training.",
    )

    # model training config
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5,
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
    )
    parser.add_argument(
        "--eval_steps", type=int, default=5000,
    )
    parser.add_argument(
        "--save_steps", type=int, default=5000,
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3,
    )
    parser.add_argument(
        "--optimizer_name", type=str, default="adamw",
    )
    parser.add_argument(
        "--output_dir", type=str, default="saved_model",
    )

    args = parser.parse_args()

    main()
