#!/bin/bash

CONFIG_PATH=configs/config.yaml

echo "Running training script..."
EXP_FOLDER=$(python src/train.py --config $CONFIG_PATH | grep "\[EXP_FOLDER\]" | sed 's/\[EXP_FOLDER\]//')