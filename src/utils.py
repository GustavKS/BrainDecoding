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
  return parser.parse_args()

def load_yaml_config(config_filename: str) -> dict:
  with open(config_filename) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
  return config

def make_exp_folder(config):
  experiment_folder = os.path.join(
    "./outputs", f"{config['experiment_folder']}"
  )
  Path(experiment_folder).mkdir(parents=True, exist_ok=True)
  with open(os.path.join(experiment_folder, "config.yaml"), "w") as f:
    yaml.dump(OmegaConf.to_container(config, resolve=True), f, default_flow_style=False)
  return experiment_folder