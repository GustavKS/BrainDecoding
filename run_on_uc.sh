#!/bin/bash

set -e
echo "Starting training..."
python src/train.py --config configs/config_unicluster.yaml
echo "Training done"