#!/bin/bash

# to train CodeT5
python3 ./model/train_T5.py --data="Celiadraw/text-to-mermaid-2" --output=./model/trained_model/ --action=train --batch_size=2 --eval_batch_size=1 # need to make batches smaller for training to not error out from memory
