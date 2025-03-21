from omegaconf import OmegaConf
import numpy as np

import torch
import wandb
from tqdm import tqdm

#from models import Model
from utils import parse_args, load_yaml_config, make_exp_folder
import dataset
from loss import CLIPLoss


if __name__ == "__main__":
  args = parse_args()
  config = load_yaml_config(config_filename=args.config)
  config = OmegaConf.create(config)
  experiment_folder = make_exp_folder(config)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  dataset = dataset.BrennanDataset(config=config)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

  if config["use_wandb"]:
    wandb.config = {k: v for k, v in config.items() if k not in ["use_wandb", "wandb_project"]}
    wandb.init(project="BrainDecoding", config=wandb.config, save_code=True)
    wandb.run.name = experiment_folder
    wandb.run.save()

  #brain_encoder = 
  #stim_encoder =

  loss_fn = CLIPLoss().to(device)

  optimizer = torch.optim.Adam(brain_encoder.parameters(), lr=1e-4)

  for epoch in range(100):
    train_losses, test_losses = []

    brain_encoder.train()

    for i, batch in enumerate(tqdm(dataloader)):
      stim, brain = batch