from omegaconf import OmegaConf
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models import naive_dec, brain_dec
from utils import parse_args, load_yaml_config, make_exp_folder
import our_dataset as our_dataset

from termcolor import cprint

if __name__ == "__main__":
  args = parse_args()
  config = load_yaml_config(config_filename=args.config)
  config = OmegaConf.create(config)
  config['experiment_folder'] = f"{config.get('experiment_folder')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
  experiment_folder = make_exp_folder(config, args.run)

  subjects = config["subjects"]

  print(f"[EXP_FOLDER]{experiment_folder}")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  writer = SummaryWriter(experiment_folder)
  datasets = []
  val_datasets = []
  for i in subjects:
    datasets.append(our_dataset.meg_dataset(config=config, s=i, transform = config['transform'], train=True))
    val_datasets.append(our_dataset.meg_dataset(config=config, s=i, transform = False, train=True))
    cprint("Subject: " + str(i) + ", Number of samples: " + str(len(datasets[-1])), "yellow")
  
  dataset = torch.utils.data.ConcatDataset(datasets)
  val_dataset = torch.utils.data.ConcatDataset(val_datasets)

  print("Expected Number of samples:", 3*50*len(subjects), "Actual Number of samples:", len(dataset))
  labels = np.array([0 if dataset[i][1] == 43 else 1 for i in range(len(dataset))])

  class_0_idx, class_1_idx = np.where(labels == 0)[0], np.where(labels == 1)[0]

  np.random.seed(42)
  np.random.shuffle(class_0_idx)
  np.random.shuffle(class_1_idx)

  val_samples_per_class = min(len(class_0_idx), len(class_1_idx)) // 5

  val_idcs = np.concatenate((class_0_idx[:val_samples_per_class], class_1_idx[:val_samples_per_class]))
  train_idcs = np.setdiff1d(np.arange(len(dataset)), val_idcs)

  train_dataset = torch.utils.data.Subset(dataset, train_idcs)
  val_dataset = torch.utils.data.Subset(val_dataset, val_idcs)
  print("Train samples:", len(train_dataset), "Validation samples:", len(val_dataset))
  
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
  val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

  backbone = torchvision.models.resnet18(weights=None)
  backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  #backbone.load_state_dict(torch.load("/home/ubuntu/BrainDecoding/outputs/ET/backbonesjlayeratttrans_resample_20250429/best_model.pth", weights_only=True), strict=False)
  #for param in backbone.parameters():
  #  param.requires_grad = False
  model = naive_dec.NaiveModel(backbone, len(subjects), config).to(device)
  #model = brain_dec.BrainDecoder(None, len(subjects), config).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=float(config['learning_rate']))
  #scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-6)
  loss_fn = nn.CrossEntropyLoss()

  val_accs = []
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

      val_accs.append(accuracy)
      if accuracy == max(val_accs):
          torch.save(model.state_dict(), f'{experiment_folder}/best_model.pth')
          torch.save({"outputs": torch.cat(all_val_outputs, dim=0), "labels": torch.cat(all_val_labels, dim=0), "subjects": torch.cat(all_subjects, dim=0)}, f'{experiment_folder}/val_outputs.pt')
          torch.save({"outputs": torch.cat(all_train_outputs, dim=0), "labels": torch.cat(all_train_labels, dim=0), "subjects": torch.cat(all_subjects, dim=0)}, f'{experiment_folder}/train_outputs.pt')
          print(f'Saved best model at epoch {epoch+1}')
      #scheduler.step()
      #writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
  writer.close()