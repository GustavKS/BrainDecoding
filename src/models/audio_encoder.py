import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(device)

        # Assuming we want the output shape [64, 1024, 360], we can add a linear layer to adjust the sequence length.
        self.linear = nn.Linear(149, 360)  # Changing the sequence length from 149 to 360

    def forward(self, x):
        if not x.dtype.is_floating_point:
            x = x.float()
        outputs = self.wav2vec(x)

        # Apply linear transformation to adjust the sequence length
        hidden_states = outputs.last_hidden_state  # Shape: [64, 149, 1024]

        # We need to adjust the shape to [64, 1024, 360]
        hidden_states = hidden_states.permute(0, 2, 1)  # Shape: [64, 1024, 149]
        hidden_states = self.linear(hidden_states)      # Shape: [64, 1024, 360]
        
        return hidden_states
