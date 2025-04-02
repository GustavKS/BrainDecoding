from omegaconf import OmegaConf
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import naive_dec, brain_enc
from utils import parse_args, load_yaml_config, make_exp_folder
import our_dataset as our_dataset

if __name__ == "__main__":
  args = parse_args()
  config = load_yaml_config(config_filename=args.config)
  config = OmegaConf.create(config)
  experiment_folder = make_exp_folder(config)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  writer = SummaryWriter(experiment_folder)
  datasets = []
  subjects = [2, 7, 10, 11]
  for i in subjects:
    datasets.append(our_dataset.meg_dataset(config=config, s=i, train=True))
  
  dataset = torch.utils.data.ConcatDataset(datasets)
  train_idcs = np.arange(0, len(dataset))
  np.random.seed(42)
  np.random.shuffle(train_idcs)
  train_idcs = train_idcs[:int(len(train_idcs)*0.8)]
  val_idcs = np.setdiff1d(np.arange(0, len(dataset)), train_idcs)
  train_dataset = torch.utils.data.Subset(dataset, train_idcs)
  val_dataset = torch.utils.data.Subset(dataset, val_idcs)

  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
  val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

  if config["naive"]:
    backbone = torchvision.models.resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = naive_dec.NaiveModel(backbone).to(device)
  else:
     model = brain_enc.BrainEncoder(subjects).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  loss_fn = nn.CrossEntropyLoss()
  val_accs = []
  for epoch in range(50):
      model.train()
      progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}', unit='batch', leave=False)
      lossinho = 0
      correct = 0
      total = 0
      for idx, (data, target, sbjs) in enumerate(progress_bar):
          target = [0 if i == 43 else 1 for i in target]
          data = data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])
          data, target = data.to(device), torch.tensor(target).to(device)
          output = model(data, sbjs)
          loss = loss_fn(output, target)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          lossinho += loss.item()

          _, predicted = torch.max(output.data, 1)
          total += target.size(0)
          correct += (predicted == target).sum().item()

          progress_bar.set_postfix(loss=lossinho/(idx+1))
      accuracy = 100 * correct / total
      print(f'Epoch {epoch+1} Loss: {lossinho/len(train_dataloader):.4f}')
      writer.add_scalar('Loss/train', lossinho/len(train_dataloader), epoch)
      writer.add_scalar('Accuracy/train', accuracy, epoch)

      model.eval()
      with torch.no_grad():
          correct = 0
          total = 0
          lossinho = 0
          for data, target, sbjs in val_dataloader:
              target = [0 if i == 43 else 1 for i in target]
              data = data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])
              data, target = data.to(device), torch.tensor(target).to(device)
              output = model(data, sbjs)
              loss = loss_fn(output, target)
              lossinho += loss.item()
              _, predicted = torch.max(output.data, 1)
              total += target.size(0)
              correct += (predicted == target).sum().item()
          accuracy = 100 * correct / total
          print(f'Accuracy: {accuracy:.2f}%, Loss: {lossinho/len(val_dataloader):.4f}')
      writer.add_scalar('Loss/val', lossinho/len(val_dataloader), epoch)
      writer.add_scalar('Accuracy/val', accuracy, epoch)

      val_accs.append(accuracy)
      if accuracy == max(val_accs):
          torch.save(model.state_dict(), f'{experiment_folder}/best_model.pth')
          print(f'Saved best model at epoch {epoch+1}')
  writer.close()