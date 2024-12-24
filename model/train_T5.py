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
from pandas.io.xml import preprocess_data
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from transformers.generation import candidate_generator
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTrainedTokenizerBase,
)

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
bleu = evaluate.load("bleu")
dataset = "Celiadraw/text-to-mermaid-2"


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
        "--model", type=str, default="google-t5/t5-small", help=" T5 model."
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


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return bleu.compute(predictions=predictions, references=labels)


def preprocess_dataset(data):
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
    ds = load_dataset(data)
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


# Data collator for seq2seq


def main():

    print("test")


if __name__ == "__main__":
    preprocess_dataset(dataset)
