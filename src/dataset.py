import glob
import scipy
import numpy as np
import torch

import torchaudio
from torch.utils.data import Dataset

class BrennanDataset(Dataset):
  def __init__(self, config):
    self.root_dir = config['root_dir']

    ## Brain Data
    mathfile_paths = glob.glob(f"{self.root_dir}/data/Brennan2018/raw/*.mat")

    for i, mathfile_path in enumerate(mathfile_paths):
      mat_raw = scipy.io.loadmat(mathfile_path)


    
    ## Audio Data
    audio_paths = glob.glob(f"{self.root_dir}/data/Brennan2018/audio/*.wav")
    waveforms = [torchaudio.load(audio_path) for audio_path in audio_paths]
    sample_rate = np.array([w[1] for w in waveforms])

    assert np.all(sample_rate == sample_rate[0]), "All audio files must have the same sample rate"

    sample_rate = sample_rate[0]

    waveforms = torch.cat([w[0] for w in waveforms], dim=1)

    print(waveforms.shape)
