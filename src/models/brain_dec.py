import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialAttention(nn.Module):
  def __init__(
    self, loc: torch.Tensor, D1: int, K: int
):
    super().__init__()

    x, y = loc.T

    self.z_re = nn.Parameter(torch.Tensor(D1, K, K))
    self.z_im = nn.Parameter(torch.Tensor(D1, K, K))
    nn.init.kaiming_uniform_(self.z_re, a=np.sqrt(5))
    nn.init.kaiming_uniform_(self.z_im, a=np.sqrt(5))

    k_arange = torch.arange(K)
    rad1 = torch.einsum("k,c->kc", k_arange, x)
    rad2 = torch.einsum("l,c->lc", k_arange, y)
    rad = rad1.unsqueeze(1) + rad2.unsqueeze(0)
    self.register_buffer("cos", torch.cos(2 * torch.pi * rad))
    self.register_buffer("sin", torch.sin(2 * torch.pi * rad))

  def forward(self, X: torch.Tensor) -> torch.Tensor:
      """_summary_

      Args:
          X ( b, c, t ): _description_

      Returns:
          _type_: _description_
      """

      real = torch.einsum("dkl,klc->dc", self.z_re, self.cos)
      imag = torch.einsum("dkl,klc->dc", self.z_im, self.sin)

      a = F.softmax(real + imag, dim=-1) 

      return torch.einsum("oi,bit->bot", a, X)


class channel_dropout(nn.Module):
    def __init__(self, num_channels: int = 268):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor, p):
        if self.training:
            batch_size, channels, time = x.shape
            dropout_mask = (torch.rand(batch_size, channels, 1, device=x.device) > p).float()
            x = x * dropout_mask
        return x

class SubjectBlock(nn.Module):
    def __init__(
        self,
        num_subjects: int,
        D1: int = 270,
        num_channels: int = 268,
    ):
        super().__init__()

        self.num_subjects = num_subjects
        self.num_channels = num_channels

        self.conv = nn.Conv1d(
            in_channels=D1,
            out_channels=D1,
            kernel_size=1,
            stride=1,
        )
        self.subject_layer = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=D1,
                    out_channels=D1,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                )
                for _ in range(self.num_subjects)
            ]
        )

    def forward(self, X: torch.Tensor, subject_idxs):
        X = self.conv(X)    
        X = torch.cat(
            [
                self.subject_layer[i](x.unsqueeze(dim=0))
                for i, x in zip(subject_idxs, X)
            ]
        )
        return X
    
class ConvBlock(nn.Module):
    def __init__(
        self,
        k: int,
        D1: int= 270,
        D2: int= 320,
        ksize: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()

        self.k = k
        in_channels = D1 if k == 0 else D2

        self.conv0 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=D2,
            kernel_size=ksize,
            padding="same",
            dilation=2 ** ((2 * self.k) % 5),
        )
        self.batchnorm0 = nn.BatchNorm1d(num_features=D2)
        self.conv1 = nn.Conv1d(
            in_channels=D2,
            out_channels=D2,
            kernel_size=ksize,
            padding="same",
            dilation=2 ** ((2 * self.k + 1) % 5),
        )
        self.batchnorm1 = nn.BatchNorm1d(num_features=D2)
        self.conv2 = nn.Conv1d(
            in_channels=D2,
            out_channels=2 * D2,
            kernel_size=ksize,
            padding="same",
            dilation=2,  # NOTE: The text doesn't say this, but the picture shows dilation=2
        )
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.k == 0:
            X = self.conv0(X)
        else:
            X = self.conv0(X) + X  # skip connection

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        X = self.conv2(X)
        X = F.glu(X, dim=-2)

        return self.dropout(X)
    
class CustomBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input):
        if input.size(0) == 1:
            return input
        return super().forward(input)

class BrainDecoder(nn.Module):
    def __init__(self, backbone: nn.Module, num_subjects: int, config):
        super().__init__()

        self.channel_dropout = config["channel_dropout"]
        self.p_channel_dropout = config["p_channel_dropout"]
        self.attention = config["attention"]
        self.subject_layer = config["subject_layer"]

        self.channel_dropout = channel_dropout(num_channels=268)

        loc = np.load("/home/ubuntu/BrainDecoding/configs/layout.npy")
        loc = torch.from_numpy(loc.astype(np.float32))
        self.spatial_attention = SpatialAttention(
            loc=loc,
            D1=270,
            K=32
        )

        self.subject_block = SubjectBlock(num_subjects=num_subjects)

        self.conv_blocks = nn.Sequential()
        for k in range(5):
            self.conv_blocks.add_module(f"conv{k}", ConvBlock(k, 270, 320))

        self.conv_final1 = nn.Conv1d(in_channels=320, out_channels=320, kernel_size=1,)
        self.conv_final2 = nn.Conv1d(in_channels=320, out_channels=370, kernel_size=1,)
        
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(370*480, 512),
            CustomBatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            CustomBatchNorm1d(128),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            CustomBatchNorm1d(64),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x, s, sbj_list):
        s_mapping = {subject: idx for idx, subject in enumerate(sbj_list)}
        s = [s_mapping[int(i)] for i in s]

        if self.attention:
            x = self.spatial_attention(x)

        if self.channel_dropout:
            x = self.channel_dropout(x, p=self.p_channel_dropout)

        if self.subject_layer:
            x = self.subject_block(x, s)
            
        x = self.conv_blocks(x)
        x = F.gelu(self.conv_final1(x))
        x = F.gelu(self.conv_final2(x))
        x = torch.flatten(x, start_dim=1)
        output = self.cls_head(x)
        return output