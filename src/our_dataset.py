import os
import re
import numpy as np
import torch
import torchvision
import mne
from sklearn.preprocessing import RobustScaler
from functools import partial
from termcolor import cprint

def scale_clamp(X: np.ndarray, clamp_lim: float = 5.0, clamp: bool = True) -> np.ndarray:
  X = X.reshape(X.shape[0], -1)
  X = RobustScaler().fit_transform(X)
  if clamp:
      X = X.clip(min=-clamp_lim, max=clamp_lim)
  return X.squeeze()

class MEGTransform(object):
  def __init__(self):
    self.meg_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), value=0),
        torchvision.transforms.RandomResizedCrop((268, 480), scale=(0.2, 1.0)),
    ])
  def __call__(self, sample):
    sample = sample.unsqueeze(0)
    sample = self.meg_transform(sample)
    sample = sample.squeeze(0)
    return sample

class meg_dataset(torch.utils.data.Dataset):
  def __init__(self, config, s: int, train: bool):
    self.meg_transform = MEGTransform()
    self.root = config['root_dir']
    nights = ['Night1', 'Night2', 'Night3', 'Night4']
    if s == 4 or s == 5:
      nights = ['Night1', 'Night3', 'Night4', 'Night5']
    np.random.seed(s)
    np.random.shuffle(nights)
    if train: 
      nights = nights[:3]
    else:
      nights = nights[3:]
    self.s = s
    s = f"S{int(s):02d}"
    mne.set_config('MNE_USE_CUDA', 'true')
    self.all_meg_data = []
    self.all_epochs = []
    cprint("Analysing Subject: " + s + ", nights: " + str(nights), 'blue', attrs=['bold'])
    for night in nights:
      print(f"Loading {s} {night}")
      data_path_folder = os.path.join(self.root, s, night)
      files = [f for f in os.listdir(data_path_folder)]
      files.sort()
      wm = files[-1]
      self.data_path = os.path.join(self.root, s, night, wm)

      raw = mne.io.read_raw_ctf(self.data_path, preload=True, verbose=False)
      raw.pick_types(meg=True, stim=True, eeg=False, ref_meg=False, verbose=False)
      if "MRP54-4016" in raw.ch_names:
        raw.drop_channels(["MRP54-4016"])
      if config["filter"]:
        raw.filter(config["filter_low"], config["filter_high"], picks='mag', n_jobs='cuda', h_trans_bandwidth=config["filter_high"]/10, l_trans_bandwidth=config["filter_low"]/10, verbose=False)
      events = mne.find_events(raw, stim_channel='UDIO001', initial_event=True, verbose=False)

      event_ids = {"maint_FACE": 43, "maint_HOUSE": 53}

      events[:,2] = events[:,2] - 256
      picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False, ref_meg=False)
      self.epochs = mne.Epochs(raw, events, event_ids,  picks=picks, tmin=0, tmax=4,\
                          baseline=None, preload=True, verbose=False)
      self.epochs.resample(120, n_jobs='cuda', verbose=False)
      self.epochs.apply_function(partial(scale_clamp, clamp_lim=config["clamp"]), n_jobs=10, verbose=False)
      if config["spectrogram"]:
        freqs = np.arange(0.25, 32.25, 0.25)
        n_cycles = freqs / 2.0
        power, _ = self.epochs.compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles, average=True, return_itc=True, picks='meg', verbose=False)
        self.meg_data = torch.from_numpy(power.data).to(torch.float32)
      else:
        self.meg_data = torch.from_numpy(self.epochs.get_data(picks='meg')).to(torch.float32)
      self.all_meg_data.append(self.meg_data)
      self.all_epochs.append(self.epochs)

    self.all_meg_data = torch.cat(self.all_meg_data, dim=0)
    for ep in self.all_epochs:
      ep.info['dev_head_t'] = self.all_epochs[0].info['dev_head_t']
    self.all_epochs = mne.concatenate_epochs(self.all_epochs, verbose=False)

    #np.random.shuffle(self.all_epochs.events[:, 2])
    
  def __len__(self):
    return len(self.all_meg_data)
  
  def __getitem__(self, idx):
    return self.meg_transform(self.all_meg_data[idx, :, :]), self.all_epochs.events[idx, 2], self.s