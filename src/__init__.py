"""PhiloGemma - A philosophical Gemma LLM trained on wisdom literature."""

from .dataloader import PhiloTextDataset, create_philo_dataloader
from .model import PhiloGemmaModel
from .trainer import PhiloGemmaTrainer

__all__ = [
    'PhiloTextDataset',
    'create_philo_dataloader',
    'PhiloGemmaModel',
    'PhiloGemmaTrainer'
]