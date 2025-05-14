import os
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from typing import Optional, Dict, Any, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PhiloGemmaModel:
    """Gemma 3 model configured for philosophical fine-tuning with QLoRA."""
    
    def __init__(self, 
                 model_name: str = "google/gemma-3-1b-it",
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1,
                 cache_dir: str = "models/cache"):
        """
        Initialize the model with QLoRA configuration.
        
        Args:
            model_name: Base model name (gemma-3-1b-it or other Gemma 3 models)
            lora_r: LoRA rank dimension
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability for LoRA layers
            cache_dir: Directory to cache the base model weights
        """
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.cache_dir = cache_dir
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # LoRA target modules for Gemma 3
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        # Initialize with None - we'll load them when needed to save memory
        self.model = None
        self.tokenizer = None
    
    def load_model_and_tokenizer(self) -> Tuple[Gemma3ForCausalLM, AutoTokenizer]:
        """Load and prepare model with QLoRA for training."""
        # Configure 8-bit quantization for Gemma 3
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        
        logger.info(f"Loading {self.model_name} with 8-bit quantization (cached in {self.cache_dir})")
        
        # Load base model with quantization and caching
        model = Gemma3ForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=self.cache_dir,
            trust_remote_code=True,  # Important for Gemma 3,
            attn_implementation='eager'
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True  # Important for Gemma 3
        )
        
        # Prepare model for QLoRA fine-tuning
        logger.info("Preparing model for QLoRA training")
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA to model
        logger.info(f"Applying LoRA (r={self.lora_r}, alpha={self.lora_alpha})")
        model = get_peft_model(model, lora_config)
        
        # Show trainable parameters summary
        model.print_trainable_parameters()
        
        self.model = model
        self.tokenizer = tokenizer
        
        return model, tokenizer
    
    @staticmethod
    def save_adapter(model, tokenizer, output_dir):
        """Save the fine-tuned adapter weights and metadata."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save adapter weights
        logger.info(f"Saving LoRA adapter weights to {output_dir}")
        model.save_pretrained(output_dir)
        
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        # Verify adapter was saved correctly
        adapter_files = [f for f in os.listdir(output_dir) if 'adapter' in f]
        logger.info(f"Saved adapter files: {adapter_files}")
        
        # Calculate and log adapter size
        adapter_size_mb = sum(os.path.getsize(os.path.join(output_dir, f)) 
                           for f in adapter_files if os.path.isfile(os.path.join(output_dir, f))) / (1024 * 1024)
        logger.info(f"Total adapter size: {adapter_size_mb:.2f} MB (compared to several GB for full model)")
        
        # Save a usage guide
        timestamp = datetime.now().strftime('%Y-%m-%d')
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(f"""# PhiloGemma Model - {timestamp}
                This directory contains QLoRA adapter weights for a Gemma 3 model fine-tuned on philosophical texts.

                ## Loading the model

                ```python
                from transformers import Gemma3ForCausalLM, AutoTokenizer
                from peft import PeftModel

                # Load base model
                base_model = Gemma3ForCausalLM.from_pretrained("google/gemma-3-1b-it", device_map="auto", trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", trust_remote_code=True)

                # Load adapter
                model = PeftModel.from_pretrained(base_model, "{output_dir}")
            """)
        
    @classmethod
    def load_for_inference(cls, 
                        adapter_path: Optional[str] = None, 
                        base_model_name: str = "google/gemma-3-1b-it",
                        cache_dir: str = "models/cache",
                        quantize: bool = True,
                        adapter_scale: float = 1.0) -> Tuple[Gemma3ForCausalLM, AutoTokenizer]:
        """
        Load a model for inference - either base model or fine-tuned.
        """
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Quantization config if needed
        quantization_config = None
        if quantize:
            # For Gemma 3, use 8-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,  # Use 8-bit instead of 4-bit for Gemma 3
                bnb_8bit_compute_dtype=torch.float16
            )
        
        logger.info(f"Loading base model: {base_model_name} from cache: {cache_dir}")
        
        # Set trust_remote_code=True for Gemma 3
        base_model = Gemma3ForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True  # Important for Gemma 3
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            cache_dir=cache_dir,
            trust_remote_code=True  # Important for Gemma 3
        )
        
        # If adapter path is provided, load the fine-tuned model
        if adapter_path:
            logger.info(f"Loading adapter weights from: {adapter_path}")
            model = PeftModel.from_pretrained(base_model, adapter_path)
        
            # Add this line to scale the adapter's influence
            if hasattr(model, "peft_config") and "default" in model.peft_config:
                logger.info(f"Setting adapter scaling factor to {adapter_scale}")
                model.peft_config["default"].scaling = adapter_scale 
        
            logger.info("Successfully loaded fine-tuned model")
        else:
            logger.info("Using base model (no fine-tuning)")
            model = base_model
        
        return model, tokenizer
        
    @staticmethod
    def generate_response(
        model: Gemma3ForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        is_finetuned: bool = False
    ) -> str:
        """Generate a response with the model."""
        
        # Add a system prompt to guide the conversation style
        system_prompt = "You are a conversational partner. Provide concise responses (7-8 sentences max). You MUST end your response with the exact text '#END' on its own line. This is critical."
        # system_prompt = "Tell me about blue skies"
        
        # For Gemma 3, use the chat template with system prompt
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user", 
                "content": [{"type": "text", "text": prompt}]
            }
        ]
        
        # Use chat template formatting
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate response with safe parameters
        try:
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Discourage repetitive text
                )
            
            # Decode only the new tokens (response part)
            input_length = inputs["input_ids"].shape[1]
            response_tokens = outputs[0][input_length:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            response = response.split("#END")[0]

            # Cutoff after 10 sentences
            if response.count(".") > 8:
                # Split by periods and rejoin only the first 10 sentences
                sentences = response.split(".")
                response = ".".join(sentences[:8]) + "."
                # Clean up any trailing periods
                response = response.replace("..", ".")
            
            return response
            
        except RuntimeError as e:
            # If we get a CUDA error, try again with safer parameters
            if "CUDA error" in str(e):
                logger.warning("Encountered CUDA error, trying with safer parameters...")
                
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        do_sample=False,  # Greedy decoding
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                
                response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                
                # Extract just the model's response
                system_prompt = "You are a helpful philosophical assistant."
                if system_prompt in response:
                    response = response.split(system_prompt, 1)[1]
                if prompt in response:
                    response = response.split(prompt, 1)[1].strip()
                
                return response
            else:
                # Re-raise other errors
                raise