#!/usr/bin/env python3
"""
Train the WattsGemma model on PDFs.

Usage:
    python train.py --input_dir data/wattstxts --output_dir models/wattsgemma
"""

import argparse
import os
import logging
from datetime import datetime
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trainer import WattsGemmaTrainer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train WattsGemma on txt files")
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/wattstxts",
        help="Directory containing files for training"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/wattsgemma",
        help="Directory to save model and logs (default: models/wattsgemma)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-1b-it",
        choices=["google/gemma-3-1b-it"],
        help="Base model name (default: google/gemma-2b)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="Batch size for training (default: 4)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank dimension (default: 16)"
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter (default: 32)"
    )
    
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="models/cache",
        help="Directory to cache base model (default: models/cache)"
    )
    
    return parser.parse_args()

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/training_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Log arguments
    logging.info(f"Training with arguments: {args}")
    
    # Check if directory exists
    if not os.path.exists(args.input_dir):
        logging.error(f"Directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize trainer
    trainer = WattsGemmaTrainer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        lora_config={
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": 0.1
        },
        max_sequence_length=args.sequence_length,
        cache_dir=args.cache_dir
    )
    
    # Train model
    try:
        output_dir = trainer.train()
        logging.info(f"Training completed. Model saved to: {output_dir}")
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()