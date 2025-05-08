subjects=(2 4 5 6 7 10 11)

for i in $(seq 1 5); do
  for subject in "${subjects[@]}"; do
    CONFIG_PATH=configs/config_"$subject".yaml
    echo "Running training script for subject $subject, run $i..."
    EXP_FOLDER=$(python src/train.py --config "$CONFIG_PATH" --run "$i" | grep "\[EXP_FOLDER\]" | sed 's/\[EXP_FOLDER\]//')
    if [ -z "$EXP_FOLDER" ]; then
      echo "ERROR: EXP_FOLDER not detected for subject $subject, run $i"
      continue
    fi
    echo "Training done. Detected experiment folder: $EXP_FOLDER"
    echo "Running testing script..."
    python src/test.py --config "$CONFIG_PATH" --root "$EXP_FOLDER"
  done
done