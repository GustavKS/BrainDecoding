from omegaconf import OmegaConf
import numpy as np

import torch

from models import Model
from utils import parse_args, load_yaml_config, make_exp_folder
from dataset import Dataset


if __name__ == "__main__":
  args = parse_args()
  config = load_yaml_config(config_filename=args.config)
  config = OmegaConf.create(config)
  experiment_folder = make_exp_folder(config)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = Model.Model().to(device)

  dataset = Dataset()