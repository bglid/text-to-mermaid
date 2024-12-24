from typing import Optional, Union

import argparse
import os
import random
from dataclasses import dataclass

import evaluate
import numpy as np
import torch
import transformers
from datasets import Dataset, DatasetDict, load_dataset
from torch.nn.modules import padding
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from transformers.convert_slow_tokenizer import Tokenizer
from transformers.generation import candidate_generator
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTrainedTokenizerBase,
)

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# dataset = "Celiadraw/text-to-mermaid-2"


def parse_args():
    parser = argparse.ArgumentParser()

    # data arg
    parser.add_argument(
        "--data", type=str, default="", help="path to training, val, or test data`"
    )

    # outputting
    parser.add_argument(
        "--output", type=str, default="", help="path to saving model checkpoints"
    )

    # model architecture
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="google-t5/t5-small",
        help="T5",
    )

    parser.add_argument(
        "--model", type=str, default="google-t5/t5-small", help=" Pretrained T5 model."
    )

    parser.add_argument(
        "--checkpoint", type=str, default="", help="path to saving model checkpoints"
    )

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # Training
    parser.add_argument(
        "--action",
        type=str,
        default="",
        choices=["train", "evaluate", "test"],
        help="Choose between training, evaluating, or testing the model.",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate...",
    )

    # potentially adjust ths and LR arg defaults incase
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )

    # Regularization
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="",
    )

    # Evaluation
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="",
    )

    # Test (think about this one)

    return parser.parse_args()


# just bleu for now
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # bleu = bleu.compute(predictions=predictions, references=labels)
    rouge.compute(predictions=predictions, references=labels)
    # return bleu
    return rouge


def dataset_split(data_to_split):
    """
    Function that takes HF dataset and splits into into train, val, and test.

    Parameters:
        data (string) : String link to the huggingface dataset.

    Returns: DatasetDict({
            "train": train_devtest["train"],
            "valid": dev_test["train"],
            "test": dev_test["test"],
    })
    """
    # load dataset
    ds = load_dataset(data_to_split)
    # splitting the dataset into train and (test + dev)
    train_devtest = ds["train"].train_test_split(test_size=0.25, seed=42)
    dev_test = train_devtest["test"].train_test_split(test_size=0.5, seed=42)

    # brining them all into one dataset
    ds_split = DatasetDict(
        {
            "train": train_devtest["train"],
            "valid": dev_test["train"],
            "test": dev_test["test"],
        }
    )

    print(f"Before:\n {ds}")
    print(f"After:\n {ds_split}")
    return ds_split


def preprocess_datasets(examples):
    # tokenizing inputs
    model_inputs = tokenizer(
        examples["prompt"], max_length=512, truncation=True, padding="max_length"
    )
    # tokenizing labels
    with tokenizer.as_target_tokenizer():
        model_labels = tokenizer(
            examples["output"], max_length=512, truncation=True, padding="max_length"
        )

    model_inputs["labels"] = model_labels["input_ids"]
    return model_inputs


def main():

    # setting seed and args
    args = parse_args()
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # opting for cuda
    device = torch.device(args.device)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(seed)
        print(f"% of GPUs available: {torch.cuda.device_count()}")
        print(f"GPU that will be used: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Running on CPU...")
        print("Killing this job")
        exit(333)

    dataset = args.data
    split = dataset_split(dataset)
    # processed dataset
    tokenized_dataset = split.map(preprocess_datasets, batched=True)
    print("DATASET EXAMPLES:")
    print(tokenized_dataset)

    # Training time:
    if args.action == "train":
        model = T5ForConditionalGeneration.from_pretrained(args.model).to(device)

        # setup early stopping
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=10, early_stopping_threshold=0.001
        )

        # Datacollator for training!
        datacollator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output),
            # report_to="wandb"
            save_strategy="epoch",
            eval_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_epochs,
            load_best_model_at_end=True,
            logging_steps=100,
            logging_first_step=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["valid"],
            processing_class=tokenizer,
            data_collator=datacollator,
            callbacks=[early_stopping],
            compute_metrics=compute_metrics,
        )

        print(f"Device: {device}")
        if device == "cuda":
            model.cuda()

        trainer.train()


if __name__ == "__main__":
    main()
