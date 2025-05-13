"""WattsGemma - A philosophical Gemma LLM trained on wisdom literature."""

from .dataloader import WattsTextDataset, create_watts_dataloader
from .model import WattsGemmaModel
from .trainer import WattsGemmaTrainer

__all__ = [
    'WattsTextDataset',
    'create_watts_dataloader',
    'WattsGemmaModel',
    'WattsGemmaTrainer'
]