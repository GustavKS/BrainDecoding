import torch
import torch.nn as nn

class CustomBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input):
        if input.size(0) == 1:
            return input
        return super().forward(input)
    
class SubjectBlock(nn.Module):
    def __init__(
        self,
        num_subjects: int,
        D1: int = 270,
        num_channels: int = 295,
    ):
        super().__init__()

        self.num_subjects = num_subjects

        self.conv = nn.Conv1d(
            in_channels=num_channels,
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

        X = self.conv(X)  # ( B, 270, 256 )
        X = torch.cat(
            [
                self.subject_layer[i](x.unsqueeze(dim=0))
                for i, x in zip(subject_idxs, X)
            ]
        )
        return X

class NaiveModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()

        self.subject_block = SubjectBlock(num_subjects=4)

        self.backbone = backbone

        out_features = list(self.backbone.modules())[-1].out_features
        
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 512),
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

    def forward(self, x, s, subject_layer:bool=True):
        if subject_layer:
            x = self.subject_block(x, s)
            print("Using subject layer")
        x = self.backbone(x)
        output = self.cls_head(x)
        return output