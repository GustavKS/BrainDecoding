CONFIG_PATH=configs/config.yaml

for i in $(seq 1 5); do
  echo "Running training script..."
  EXP_FOLDER=$(python src/train.py --config $CONFIG_PATH | grep "\[EXP_FOLDER\]" | sed 's/\[EXP_FOLDER\]//')

  echo "Training done. Detected experiment folder: $EXP_FOLDER"
  echo "Running testing script..."

  python src/test.py --config $CONFIG_PATH --root "$EXP_FOLDER"
done