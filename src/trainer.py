import os
import json
import torch
from transformers import Trainer, TrainingArguments
from typing import Dict, Any, Optional, Union
import logging
from datetime import datetime

from .dataloader import create_philo_dataloader
from .model import PhiloGemmaModel

logger = logging.getLogger(__name__)

class PhiloGemmaTrainer:
    """Trainer for the Philo Gemma model."""
    
    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 model_name: str = "google/gemma-2b",
                 batch_size: int = 4,
                 learning_rate: float = 1e-4,
                 num_epochs: int = 3,
                 gradient_accumulation_steps: int = 4,
                 lora_config: Optional[Dict[str, Any]] = None,
                 max_sequence_length: int = 512,
                 cache_dir: str = "models/cache"):
        """
        Initialize trainer for PhiloGemma.
        
        Args:
            input_dir: Directory containing training files
            output_dir: Directory to save model and logs
            model_name: Base model name
            batch_size: Batch size for training
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            gradient_accumulation_steps: Steps for gradient accumulation
            lora_config: Configuration for LoRA (if None, use defaults)
            max_sequence_length: Maximum sequence length for tokenizer
            cache_dir: Directory to cache base model
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_sequence_length = max_sequence_length
        self.cache_dir = cache_dir
        
        # Initialize LoRA config
        lora_config = lora_config or {}
        self.lora_r = lora_config.get("r", 16)
        self.lora_alpha = lora_config.get("alpha", 32)
        self.lora_dropout = lora_config.get("dropout", 0.1)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging to file
        self._setup_logging()
        
        logger.info(f"Initialized PhiloGemmaTrainer with: "
                   f"model={model_name}, batch_size={batch_size}, "
                   f"lr={learning_rate}, epochs={num_epochs}")
    
    def _setup_logging(self):
        """Set up logging to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.output_dir, f"training_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add file handler to root logger
        logging.getLogger().addHandler(file_handler)
    
    def train(self):
        """Train the model on philosophical texts."""
        # 1. Initialize model
        gemma_model = PhiloGemmaModel(
            model_name=self.model_name,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            cache_dir=self.cache_dir
        )
        
        model, tokenizer = gemma_model.load_model_and_tokenizer()
        
        # 2. Create dataloader
        logger.info(f"Creating dataloader for files in {self.input_dir}")
        dataloader, dataset = create_philo_dataloader(
            input_dir=self.input_dir,
            tokenizer=tokenizer,
            batch_size=self.batch_size,
            max_length=self.max_sequence_length
        )
        
        # 3. Setup training arguments
        run_name = f"philogemma_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Uses learning rate linear decay naturally
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_epochs,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=50,
            save_strategy="epoch",
            fp16=True,  # Use mixed precision
            report_to="tensorboard",
            run_name=run_name,
            weight_decay=0.003,
            # gradient_checkpointing=True,
            # gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        
        # 4. Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=lambda data: {
                'input_ids': torch.stack([f['input_ids'] for f in data]),
                'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                'labels': torch.stack([f['labels'] for f in data])
            }
        )
        
        # 5. Train model
        logger.info("Starting training...")
        trainer.train()
        
        # 6. Save model
        logger.info(f"Saving model to {self.output_dir}")
        PhiloGemmaModel.save_adapter(model, tokenizer, self.output_dir)
        
        # 7. Save training metadata
        self._save_training_metadata(dataset)
        
        logger.info("Training completed successfully")
        return self.output_dir
    
    def _save_training_metadata(self, dataset):
        """Save metadata about the training run."""
        unique_sources = set(doc["source"] for doc in dataset.documents)
    
        metadata = {
            "model_name": self.model_name,
            "training_date": datetime.now().isoformat(),
            "num_documents": len(dataset.documents),
            "num_chunks": len(dataset),
            "sources": list(unique_sources),
            "training_params": {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "max_sequence_length": self.max_sequence_length,
            }
        }
        
        with open(os.path.join(self.output_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)