#!/usr/bin/env python3
"""
Interactive CLI for WattsGemma model.

This script provides a command-line interface for interacting with either the base
Gemma model or the fine-tuned WattsGemma model.

Usage:
    # Use base model
    python inference.py
    
    # Use fine-tuned model
    python inference.py --model_path models/wattsgemma
"""

import argparse
import os
import sys
import logging
from datetime import datetime

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import WattsGemmaModel

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Interactive CLI for WattsGemma")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model (default: None, uses base model)"
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/gemma-3-1b-it",
        choices=["google/gemma-3-1b-it"],
        help="Base model name (default: google/gemma-3-1b-it)"
    )
    
    # Add this new argument
    parser.add_argument(
        "--adapter_scale",
        type=float,
        default=1.0,
        help="Scaling factor for adapter influence (default: 1.0, higher = more stylized)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum response length (default: 512)"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter (default: 0.9)"
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="models/cache",
        help="Directory to cache base model (default: models/cache)"
    )
    
    parser.add_argument(
        "--no_quantize",
        action="store_true",
        help="Disable model quantization (uses more memory but may be more accurate)"
    )
    
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to log file (default: None)"
    )
    
    return parser.parse_args()

def setup_logging(log_file=None):
    """Set up logging."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def print_header(model_path, base_model):
    """Print welcome header."""
    print("\n" + "=" * 80)
    print(f"{'WattsGemma AI':^80}")
    print("=" * 80)
    
    if model_path:
        print(f"Using fine-tuned model: {model_path}")
        print(f"Base model: {base_model}")
    else:
        print(f"Using base model: {base_model} (no fine-tuning)")
    
    print("\nType your questions below.")
    print("Type 'exit', 'quit', or press Ctrl+C to end the conversation.")
    print("=" * 80 + "\n")

def main():
    """Main interactive CLI function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    if args.log_file:
        log_file = args.log_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "logs/inference"
        os.makedirs(log_dir, exist_ok=True)
        log_file = f"{log_dir}/inference_{timestamp}.log"
    
    setup_logging(log_file)
    
    # Log arguments
    logging.info(f"Starting inference with arguments: {args}")
    
    # Check if model path exists if specified
    if args.model_path and not os.path.exists(args.model_path):
        logging.error(f"Model path not found: {args.model_path}")
        print(f"Error: Model path not found: {args.model_path}")
        sys.exit(1)
    
    # Print welcome header
    print_header(args.model_path, args.base_model)
    
    try:
        # Load model
        logging.info("Loading model...")
        print("Loading model, please wait...")
        
        model, tokenizer = WattsGemmaModel.load_for_inference(
            adapter_path=args.model_path,
            base_model_name=args.base_model,
            cache_dir=args.cache_dir,
            quantize=not args.no_quantize,
            adapter_scale=args.adapter_scale
        )
        
        logging.info("Model loaded successfully")
        print("Model loaded and ready for conversation!\n")
        
        # Interactive loop
        conversation_history = []
        
        while True:
            # Get user input
            try:
                user_input = input("\n> ")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting...")
                break
            
            if not user_input.strip():
                continue
            
            # Log user input
            logging.info(f"User: {user_input}")
            conversation_history.append(f"User: {user_input}")
            
            # Generate response
            try:
                print("\nThinking...")
                response = WattsGemmaModel.generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=user_input,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    is_finetuned=args.model_path is not None
                )
                
                # Log and print response
                # logging.info(f"WattsGemma: {response}")
                conversation_history.append(f"WattsGemma: {response}")
                
                print(f"\n{response}")
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logging.error(error_msg, exc_info=True)
                print(f"\nError: {error_msg}")
        
        # Save conversation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conversation_dir = "conversations"
        os.makedirs(conversation_dir, exist_ok=True)
        
        with open(f"{conversation_dir}/conversation_{timestamp}.txt", "w") as f:
            f.write("\n\n".join(conversation_history))
        
        print(f"\nConversation saved to: {conversation_dir}/conversation_{timestamp}.txt")
        
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()