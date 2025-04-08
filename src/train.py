from omegaconf import OmegaConf
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models import naive_dec, brain_enc
from utils import parse_args, load_yaml_config, make_exp_folder
import our_dataset as our_dataset

from termcolor import cprint

if __name__ == "__main__":
  subjects = [2, 4, 5, 6, 7, 10, 11]

  args = parse_args()
  config = load_yaml_config(config_filename=args.config)
  config = OmegaConf.create(config)
  config['experiment_folder'] = f"{config.get('experiment_folder')}_{str(len(subjects))}sbjs_drop{config.get('p_channel_dropout')}_{datetime.now().strftime("%Y%m%d")}"
  experiment_folder = make_exp_folder(config)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  writer = SummaryWriter(experiment_folder)
  datasets = []
  
  for i in subjects:
    datasets.append(our_dataset.meg_dataset(config=config, s=i, train=True))
    cprint("Subject: " + str(i) + ", Number of samples: " + str(len(datasets[-1])), "yellow")
  
  dataset = torch.utils.data.ConcatDataset(datasets)
  print("Expected Number of samples:", 3*400*len(subjects), "Actual Number of samples:", len(dataset))
  train_idcs = np.arange(0, len(dataset))
  np.random.seed(42)
  np.random.shuffle(train_idcs)
  train_idcs = train_idcs[:int(len(train_idcs)*0.8)]
  val_idcs = np.setdiff1d(np.arange(0, len(dataset)), train_idcs)
  train_dataset = torch.utils.data.Subset(dataset, train_idcs)
  val_dataset = torch.utils.data.Subset(dataset, val_idcs)

  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
  val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

  backbone = torchvision.models.resnet18(weights=None)
  backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  model = naive_dec.NaiveModel(backbone, len(subjects), config).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=float(config['learning_rate']))
  scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-6)
  loss_fn = nn.CrossEntropyLoss()

  val_losses = []
  for epoch in range(config['num_epochs']):
      model.train()
      progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}', unit='batch', leave=False)
      lossinho = 0
      correct = 0
      total = 0
      all_train_outputs, all_train_labels, all_subjects = [], [], []
      for idx, (data, target, sbjs) in enumerate(progress_bar):
          target = [0 if i == 43 else 1 for i in target]
          data, target = data.to(device), torch.tensor(target).to(device)
          output = model(data, sbjs, subjects)
          loss = loss_fn(output, target)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          lossinho += loss.item()

          _, predicted = torch.max(output.data, 1)
          total += target.size(0)
          correct += (predicted == target).sum().item()

          all_train_outputs.append(output.cpu())
          all_train_labels.append(target.cpu())
          all_subjects.append(sbjs.cpu())
          progress_bar.set_postfix(loss=lossinho/(idx+1))
      accuracy = 100 * correct / total
      print(f'Epoch {epoch+1} Loss: {lossinho/len(train_dataloader):.4f}')
      writer.add_scalar('Loss/train', lossinho/len(train_dataloader), epoch)
      writer.add_scalar('Accuracy/train', accuracy, epoch)

      model.eval()
      all_val_outputs, all_val_labels, all_subjects = [], [], []
      with torch.no_grad():
          correct = 0
          total = 0
          lossinho = 0
          for data, target, sbjs in val_dataloader:
              target = [0 if i == 43 else 1 for i in target]
              data, target = data.to(device), torch.tensor(target).to(device)
              output = model(data, sbjs, subjects)
              loss = loss_fn(output, target)
              lossinho += loss.item()
              _, predicted = torch.max(output.data, 1)
              total += target.size(0)
              correct += (predicted == target).sum().item()
              all_val_outputs.append(output.cpu())
              all_val_labels.append(target.cpu())
              all_subjects.append(sbjs.cpu())
          accuracy = 100 * correct / total
          print(f'Accuracy: {accuracy:.2f}%, Loss: {lossinho/len(val_dataloader):.4f}')
      writer.add_scalar('Loss/val', lossinho/len(val_dataloader), epoch)
      writer.add_scalar('Accuracy/val', accuracy, epoch)

      val_losses.append(lossinho/len(val_dataloader))
      if lossinho/len(val_dataloader) <= min(val_losses):
          torch.save(model.state_dict(), f'{experiment_folder}/best_model.pth')
          torch.save({"outputs": torch.cat(all_val_outputs, dim=0), "labels": torch.cat(all_val_labels, dim=0), "subjects": torch.cat(all_subjects, dim=0)}, f'{experiment_folder}/val_outputs.pt')
          torch.save({"outputs": torch.cat(all_train_outputs, dim=0), "labels": torch.cat(all_train_labels, dim=0), "subjects": torch.cat(all_subjects, dim=0)}, f'{experiment_folder}/train_outputs.pt')
          print(f'Saved best model at epoch {epoch+1}')
      scheduler.step()
      writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
  writer.close()