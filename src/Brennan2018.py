import glob
import scipy
import numpy as np
import torch
import mne
from natsort import natsorted

import torchaudio
from torch.utils.data import Dataset

class BrennanDataset(Dataset):
  def __init__(self, config):
    self.root_dir = config['root_dir']

    ## Brain Data
    mathfile_paths = glob.glob(f"{self.root_dir}/data/Brennan2018/raw/*.mat")[0:1]

    L = []
    for i, mathfile_path in enumerate(mathfile_paths):
      mat_raw = scipy.io.loadmat(mathfile_path)["raw"][0, 0]
      eeg_raw = mat_raw["trial"][0, 0][:60]
      L.append(eeg_raw.shape)
    trim_eeg = np.stack(L)[:, 1].flatten().min()

    eeg = []
    for i, mathfile_path in enumerate(mathfile_paths):
      mat_raw = scipy.io.loadmat(mathfile_path)["raw"][0, 0]
      eeg_raw = mat_raw["trial"][0, 0][:60, :trim_eeg]
      fsample = mat_raw["fsample"][0, 0]
      eeg_filtered = mne.filter.filter_data(eeg_raw, fsample, 1, 60)
      eeg_resampled = torchaudio.transforms.Resample(orig_freq=fsample, new_freq=120)(torch.tensor(eeg_filtered).float())
      eeg.append(eeg_resampled)
    self.eeg = torch.stack(eeg)
    self.num_subjects = self.eeg.shape[0]


    
    ## Audio Data
    audio_paths = natsorted(glob.glob(f"{self.root_dir}/data/Brennan2018/audio/*.wav"))
    waveforms = [torchaudio.load(audio_path) for audio_path in audio_paths]
    sample_rate = np.array([w[1] for w in waveforms])

    assert np.all(sample_rate == sample_rate[0]), "All audio files must have the same sample rate"

    sample_rate = sample_rate[0]

    waveform = torch.cat([w[0] for w in waveforms], dim=1)
    

    new_sample_rate = 16000
    self.audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)(waveform)

    print("Audio time in minutes: ", self.audio.shape[-1] / new_sample_rate / 60, "EEG time in minutes: ", self.eeg.shape[-1] / 120 / 60)

    ## Segmenting EEG and audio into 3-second windows
    num_segments_eeg = self.eeg.shape[-1] // 3 * 120
    num_segments_audio = self.audio.shape[-1] // 3 * 16000

    ## Trim and split into segments
    self.eeg = self.eeg[..., : num_segments_eeg * 3 * 120].split(3 * 120, dim=-1)[:240]
    self.audio = self.audio[..., : num_segments_audio * 3 * 16000].split(3 * 16000, dim=-1)[:240]

    ## Flatten the lists to store individual segments
    self.eeg_segments = list(self.eeg)
    self.audio_segments = list(self.audio)

    print(f"Total EEG segments: {len(self.eeg_segments)}, Total Audio segments: {len(self.audio_segments)}")



  def __len__(self):
    return len(self.eeg_segments)
  
  def __getitem__(self, idx):
    return self.audio_segments[idx], self.eeg_segments[idx], np.random.randint(0, self.num_subjects, 1)[0]