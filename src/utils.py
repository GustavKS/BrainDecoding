import argparse
import yaml
from pathlib import Path
from omegaconf import OmegaConf
import os

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-c",
    "--config",
    type=str,
    default="configs/config.yaml",
  )
  parser.add_argument('--root', type=str, required=False, default=None)
  parser.add_argument('--run', type=int, required=False, default=0)
  return parser.parse_args()

def load_yaml_config(config_filename: str) -> dict:
  with open(config_filename) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
  return config

def make_exp_folder(config, run):
    parts = os.path.normpath(config['experiment_folder']).split(os.sep)

    if parts[0] == 'outputs':
        parts = parts[1:]

    if parts and not parts[0].startswith('base'):
        parts[0] = f"{parts[0]}_{run}"

    experiment_folder = os.path.join('./outputs', *parts)
    Path(experiment_folder).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(experiment_folder, "config.yaml"), "w") as f:
        yaml.dump(OmegaConf.to_container(config, resolve=True), f, default_flow_style=False)

    return experiment_folder