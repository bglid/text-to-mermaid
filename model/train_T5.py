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


def parse_args():
    parser = argparse.ArgumentParser()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return bleu.compute(predictions=predictions, references=labels)


def main():
    print("test")


if __name__ == "__main__":
    preprocess_dataset(dataset)
