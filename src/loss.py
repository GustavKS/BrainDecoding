import torch
import torch.nn as nn


class CLIPLoss(nn.Module):
  def __init__(self):
    super(CLIPLoss, self).__init__()
    self.CE = nn.CrossEntropyLoss(reduction='mean')
    self.temperature = 0.07

  def forward(self, stim_emb, brain_emb):
    stim_emb = stim_emb.reshape(stim_emb.shape[0], -1)
    brain_emb = brain_emb.reshape(brain_emb.shape[0], -1)
    
    targets = torch.arange(stim_emb.shape[0]).to(stim_emb.device)

    ## To Dos: - Normalize embeddings

    sim = torch.matmul(stim_emb, brain_emb.T)
    sim /= self.temperature
    loss = (self.CE(sim, targets) + self.CE(sim.T, targets)) / 2
    return loss