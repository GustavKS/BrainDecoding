import sys
import numpy as np
import torch
import torch.nn as nn
from time import time
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
import mne
import mne_bids

def ch_locations_2d():
    montage = mne.channels.make_standard_montage("easycap-M10")
    info = mne.create_info(ch_names=montage.ch_names, sfreq=512., ch_types="eeg")
    info.set_montage(montage)

    layout = mne.channels.find_layout(info, ch_type="eeg")

    loc = layout.pos[:, :2]  # ( 61, 2 )

    loc = np.delete(loc, 28, axis=0)  # ( 60, 2 )

    # min-max normalization
    loc = (loc - loc.min(axis=0)) / (loc.max(axis=0) - loc.min(axis=0))

    # NOTE: "In practice, as a_j is periodic, we scale down (x,y) to keep a margin of 0.1 on each side."
    loc = loc * 0.8 + 0.1

    return torch.from_numpy(loc.astype(np.float32))

class SpatialAttention(nn.Module):
    """Same as SpatialAttentionVer2, but a little more concise"""

    def __init__(self, K, D1):
        super(SpatialAttention, self).__init__()

        # vectorize of k's and l's
        a = []
        for k in range(K):
            for l in range(K):
                a.append((k, l))
        a = torch.tensor(a)
        k, l = a[:, 0], a[:, 1]

        # vectorize x- and y-positions of the sensors
        loc = ch_locations_2d()
        x, y = loc[:, 0], loc[:, 1]

        # make a complex-valued parameter, reshape k,l into one dimension
        self.z = nn.Parameter(torch.rand(size=(D1, K ** 2), dtype=torch.cfloat))

        # NOTE: pre-compute the values of cos and sin (they depend on k, l, x and y which repeat)
        phi = (
            2 * torch.pi * (torch.einsum("k,x->kx", k, x) + torch.einsum("l,y->ly", l, y))
        )  # torch.Size([1024, 60]))
        self.register_buffer('cos', torch.cos(phi))
        self.register_buffer('sin', torch.sin(phi))

    def forward(self, X):
        """X: (batch_size, num_channels, T)"""

        # NOTE: do hadamard product and and sum over l and m (i.e. m, which is l X m)
        re = torch.einsum("jm, me -> je", self.z.real, self.cos)  # torch.Size([270, 60])
        im = torch.einsum("jm, me -> je", self.z.imag, self.sin)
        a = (
            re + im
        )  # essentially (unnormalized) weights with which to mix input channels into ouput channels
        # ( D1, num_channels )

        # NOTE: to get the softmax spatial attention weights over input electrodes,
        # we don't compute exp, etc (as in the eq. 5), we take softmax instead:
        SA_wts = F.softmax(a, dim=-1)  # each row sums to 1
        # ( D1, num_channels )

        # NOTE: each output is a diff weighted sum over each input channel
        return torch.einsum("oi,bit->bot", SA_wts, X)


class SubjectBlock(nn.Module):
    def __init__(self, num_subjects, D1, K):
        super(SubjectBlock, self).__init__()

        self.num_subjects = num_subjects
        self.D1 = D1
        self.K = K
        self.spatial_attention = SpatialAttention(self.K, self.D1)
        self.conv = nn.Conv1d(in_channels=self.D1, out_channels=self.D1, kernel_size=1, stride=1)
        self.subject_layer = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.D1,
                    out_channels=self.D1,
                    kernel_size=1,
                    bias=False,
                    stride=1
                )
                for _ in range(self.num_subjects)
            ]
        )

    def forward(self, X, subject_idxs):
        X = self.spatial_attention(X)  # ( B, 270, 256 )
        X = self.conv(X)  # ( B, 270, 256 )
        X = torch.cat(
            [self.subject_layer[i](x.unsqueeze(dim=0)) for i, x in zip(subject_idxs, X)]
        )  # ( B, 270, 256 )
        return X


class ConvBlock(nn.Module):
    def __init__(self, k, D1, D2):
        super(ConvBlock, self).__init__()

        self.k = k
        self.D2 = D2
        self.in_channels = D1 if k == 0 else D2

        self.conv0 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.D2,
            kernel_size=3,
            padding="same",
            dilation=2 ** ((2 * k) % 5),
        )
        self.batchnorm0 = nn.BatchNorm1d(num_features=self.D2)
        self.conv1 = nn.Conv1d(
            in_channels=self.D2,
            out_channels=self.D2,
            kernel_size=3,
            padding="same",
            dilation=2 ** ((2 * k + 1) % 5),
        )
        self.batchnorm1 = nn.BatchNorm1d(num_features=self.D2)
        self.conv2 = nn.Conv1d(
            in_channels=self.D2,
            out_channels=2 * self.D2,
            kernel_size=3,
            padding="same",
            dilation=2,  # FIXME: The text doesn't say this, but the picture shows dilation=2
        )

    def forward(self, X):
        if self.k == 0:
            X = self.conv0(X)
        else:
            X = self.conv0(X) + X  # skip connection

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        X = self.conv2(X)
        X = F.glu(X, dim=-2)

        return X  # ( B, 320, 256 )


class BrainEncoder(nn.Module):
    def __init__(self, num_subjects):
        super(BrainEncoder, self).__init__()

        self.num_subjects = num_subjects
        self.D1 = 270
        self.D2 = 320
        self.F = 1024
        self.K = 32

        self.subject_block = SubjectBlock(self.num_subjects, self.D1, self.K)
        # self.subject_block = SubjectBlock_proto(args)
        # cprint("USING THE OLD IMPLEMENTATION OF THE SUBJECT BLOCK", 'red', 'on_blue', attrs=['bold'])

        self.conv_blocks = nn.Sequential()
        for k in range(5):
            self.conv_blocks.add_module(f"conv{k}", ConvBlock(k, self.D1, self.D2))

        self.conv_final1 = nn.Conv1d(in_channels=self.D2, out_channels=2 * self.D2, kernel_size=1,)
        self.conv_final2 = nn.Conv1d(in_channels=2 * self.D2, out_channels=self.F, kernel_size=1,)

    def forward(self, X, subject_idxs):
        X = self.subject_block(X, subject_idxs)
        X = self.conv_blocks(X)
        X = F.gelu(self.conv_final1(X))
        X = F.gelu(self.conv_final2(X))
        return X