import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import Dict, List, Tuple

class WattsTextDataset(Dataset):
    """Dataset for Alan Watts texts"""
    
    def __init__(self, 
                 input_dir: str,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 512,
                 stride: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        # Extract text from files
        self.documents = self._load_documents(input_dir)
        
        # Create tokenized chunks
        self.examples = self._create_examples()
    
    def _load_documents(self, input_dir: str) -> List[Dict]:
        """Load all documents from the directory."""
        documents = []
        
        text_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.txt')]
        
        for filename in text_files:
            file_path = os.path.join(input_dir, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                
                # Clean the text
                text = re.sub(r'\s+', ' ', text).strip()
                
                documents.append({
                    "source": filename,
                    "text": text
                })
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        
        return documents
    
    def _create_examples(self) -> List[Dict]:
        """Create examples with fixed length."""
        examples = []
        
        for doc in self.documents:
            # Tokenize the document
            encodings = self.tokenizer(doc["text"], add_special_tokens=False)
            input_ids = encodings["input_ids"]
            length = len(input_ids)
            
            # Create examples with stride
            for i in range(0, length, self.stride):
                # Skip last small chunk
                if i > 0 and i + self.max_length >= length and length - i < self.max_length // 2:
                    continue
                
                # Extract chunk (without special tokens)
                chunk_end = min(i + self.max_length - 2, length)  # Reserve space for special tokens
                chunk_ids = input_ids[i:chunk_end]
                
                # Add special tokens
                model_input_ids = self.tokenizer.build_inputs_with_special_tokens(chunk_ids)
                
                # Force to exact length by padding or truncating
                padded_input_ids = self._pad_or_truncate(model_input_ids)
                attention_mask = torch.ones(self.max_length, dtype=torch.long)
                
                # Mark padding positions in attention mask
                if len(model_input_ids) < self.max_length:
                    attention_mask[len(model_input_ids):] = 0
                
                examples.append({
                    "source": doc["source"],
                    "input_ids": padded_input_ids,
                    "attention_mask": attention_mask,
                })
        
        return examples
    
    def _pad_or_truncate(self, input_ids):
        """Ensure input_ids are exactly max_length by padding or truncating."""
        if len(input_ids) > self.max_length:
            # Truncate
            return torch.tensor(input_ids[:self.max_length])
        elif len(input_ids) < self.max_length:
            # Pad
            padding = [self.pad_token_id] * (self.max_length - len(input_ids))
            return torch.tensor(input_ids + padding)
        else:
            # Already correct length
            return torch.tensor(input_ids)
    
    def __len__(self):
        """Return the number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get an example."""
        item = self.examples[idx]
        
        # For causal language modeling, labels are input_ids with padding ignored
        labels = item["input_ids"].clone()
        # Set padding tokens to -100 so they're ignored in loss calculation
        labels[item["attention_mask"] == 0] = -100
        
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": labels
        }

def create_watts_dataloader(
    input_dir: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    stride: int = 128,
    num_workers: int = 2
) -> Tuple[DataLoader, WattsTextDataset]:
    """Create dataloader for Alan Watts text dataset."""
    
    # Create dataset with fixed-length examples
    dataset = WattsTextDataset(
        input_dir=input_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride
    )
    
    # No custom collate_fn needed since all examples are already same size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    return dataloader, dataset