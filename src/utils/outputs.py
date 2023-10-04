from dataclasses import dataclass
from transformers.utils import ModelOutput
import torch

@dataclass
class BiEncoderOutput(ModelOutput):
    output: torch.FloatTensor = None
    dialogue_attentions: torch.FloatTensor = None
    desc_attentions: torch.FloatTensor = None

@dataclass
class AttentionBiEncoderOutput(ModelOutput):
    output: torch.FloatTensor = None
    dialogue_attentions: torch.FloatTensor = None
    desc_attentions: torch.FloatTensor = None
    cross_attentions: torch.FloatTensor = None