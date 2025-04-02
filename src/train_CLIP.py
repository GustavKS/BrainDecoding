from omegaconf import OmegaConf
import numpy as np

import torch
import wandb
from tqdm import tqdm

from models import img_encoder
from models import brain_enc
from utils import parse_args, load_yaml_config, make_exp_folder
import src.Brennan2018 as Brennan2018
from loss import CLIPLoss


if __name__ == "__main__":
  args = parse_args()
  config = load_yaml_config(config_filename=args.config)
  config = OmegaConf.create(config)
  experiment_folder = make_exp_folder(config)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  ds = Brennan2018.BrennanDataset(config=config)

  dataloader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=True)

  if config["use_wandb"]:
    wandb.config = {k: v for k, v in config.items() if k not in ["use_wandb", "wandb_project"]}
    wandb.init(project="BrainDecoding", config=wandb.config, save_code=True)
    wandb.run.name = experiment_folder
    wandb.run.save()

  brain_encoder = brain_enc.BrainEncoder(ds.num_subjects).to(device)
  stim_encoder = img_encoder.ImageEncoder().to(device)

  loss_fn = CLIPLoss().to(device)

  optimizer = torch.optim.Adam(list(brain_encoder.parameters()) + list(stim_encoder.parameters()), lr=1e-4)

  if config['no_stim_training']:
    for param in stim_encoder.parameters():
      param.requires_grad = False

  for epoch in range(100):
    train_losses, test_losses = [], []
 
    brain_encoder.train()

    for i, batch in enumerate(tqdm(dataloader)):
      stim, brain, subj_idx = batch
      stim, brain = stim.squeeze(1), brain.squeeze(1)
      stim, brain = stim.to(device), brain.to(device)


      Z_stim = stim_encoder(stim)
      Z_brain = brain_encoder(brain, subj_idx)

      loss = loss_fn(Z_stim, Z_brain)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

    if epoch % 2 == 0:
      print(f"Epoch: {epoch}, Train Loss: {np.mean(train_losses)}")