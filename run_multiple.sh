#!/bin/bash

## Run1

CONFIG_PATH=configs/config_noattention.yaml

echo "Running training script..."
EXP_FOLDER=$(python src/train.py --config $CONFIG_PATH | grep "\[EXP_FOLDER\]" | sed 's/\[EXP_FOLDER\]//')

echo "Training done. Detected experiment folder: $EXP_FOLDER"
echo "Running testing script..."

python src/test.py --config $CONFIG_PATH --root "$EXP_FOLDER"

## Run2

CONFIG_PATH=configs/config_nodropout.yaml

echo "Running training script..."
EXP_FOLDER=$(python src/train.py --config $CONFIG_PATH | grep "\[EXP_FOLDER\]" | sed 's/\[EXP_FOLDER\]//')

echo "Training done. Detected experiment folder: $EXP_FOLDER"
echo "Running testing script..."

python src/test.py --config $CONFIG_PATH --root "$EXP_FOLDER"

## Run3

CONFIG_PATH=configs/config_nosubjlayer.yaml

echo "Running training script..."
EXP_FOLDER=$(python src/train.py --config $CONFIG_PATH | grep "\[EXP_FOLDER\]" | sed 's/\[EXP_FOLDER\]//')

echo "Training done. Detected experiment folder: $EXP_FOLDER"
echo "Running testing script..."

python src/test.py --config $CONFIG_PATH --root "$EXP_FOLDER"

## Run4

CONFIG_PATH=configs/config.yaml

echo "Running training script..."
EXP_FOLDER=$(python src/train.py --config $CONFIG_PATH | grep "\[EXP_FOLDER\]" | sed 's/\[EXP_FOLDER\]//')

echo "Training done. Detected experiment folder: $EXP_FOLDER"
echo "Running testing script..."

python src/test.py --config $CONFIG_PATH --root "$EXP_FOLDER"