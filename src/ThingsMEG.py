import os
from functools import partial

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
from PIL import Image
from torchvision import transforms
import mne


def scale_clamp(
    X: np.ndarray,
    clamp_lim: float = 5.0,
    clamp: bool = True,
    scale_transposed: bool = True,
) -> np.ndarray:
    
    X = RobustScaler().fit_transform(X.T if scale_transposed else X)
    if scale_transposed:
        X = X.T
    if clamp:
        X = X.clip(min=-clamp_lim, max=clamp_lim)
    return X

class ImgTransform:
  def __init__(self):
    self.transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

  def __call__(self, img):
    return self.transform(img)


class ThingsMEGDataset(torch.utils.data.Dataset):
  def __init__(self, config):
    self.config = config
    self.transform = ImgTransform()
    meg_paths = [
      os.path.join(config['meg_dir'], f"preprocessed_P{i+1}-epo.fif") for i in range(4)
    ]

    sample_attrs_paths = [
      os.path.join(config['thingsmeg_root'], f"sourcedata/sample_attributes_P{i+1}.csv") for i in range(4)
    ]

    megs = []
    imgs = []
    subjects_ids = []
    categories = []
    for subject_id, (meg_path, sample_attrs_path) in enumerate(zip(meg_paths, sample_attrs_paths)):
      sample_attrs = pd.read_csv(sample_attrs_path)

      ## Images
      imgs_id = []
      for path in sample_attrs[:, 8]:
        if "images_meg" in path:
          imgs_id.append(
            os.path.join(config['img_dir'], "/".join(path.split("/")[1:]))
          )
        elif "images_test_meg" in path:
          imgs_id.append(
            os.path.join(config['img_dir'], "_".join(path.split("_")[:-1]), os.path.basename(path))
          )
        elif "images_catch_meg" in path:
          imgs_id.append(
            os.path.join(config['img_dir'], "black.jpg"))
        else:
          raise ValueError(f"Invalid image path: {path}")
      imgs.append(imgs_id)
      

      ## MEG data
      epochs = mne.read_epochs(meg_path, preload=True)
      epochs.resample(120, n_jobs=8)
      epochs.apply_function(
        partial(scale_clamp), scale_transposed=False, clamp_lim=5.0
      )
      epochs.apply_baseline((None, 0))

      meg_data = torch.from_numpy(epochs.get_data()).to(torch.float32)
      megs.append(meg_data)

      categories.append(torch.from_numpy(sample_attrs[:,2].astype(int)))
      subjects_ids.append(
         torch.ones(len(sample_attrs), dtype=int) * subject_id
      )

    self.meg = torch.cat(megs, dim=0)
    self.imgs = torch.cat(imgs, dim=0)
    self.subject_ids = torch.cat(subjects_ids, dim=0)
    self.categories = torch.cat(categories) - 1
    self.num_categories = len(torch.unique(self.categories))

  def __len__(self):
    return len(self.meg)
  
  def __getitem__(self, idx):
     img_path = self.imgs[idx]
     img = Image.open(img_path).convert("RGB")
     image = self.transform(img)
     return self.meg[idx], image, self.subject_ids[idx], self.categories[idx]


        
    
