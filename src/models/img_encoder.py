import torch
import torch.nn as nn
from transformers import CLIPVisionModel


class ImageEncoder(nn.Module):
  def __init__(self, ):
    super().__init__()
    self.img_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").visual

  def forward(self, x):
    x = self.img_encoder(x)
    return x