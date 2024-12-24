#!/bin/bash

# to train CodeT5
python3 ./model/train_T5.py --data="Celiadraw/text-to-mermaid-2" --output=./model/trained_model/ --action=train
