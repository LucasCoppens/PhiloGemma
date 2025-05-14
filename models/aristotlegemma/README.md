# PhiloGemma Model - 2025-05-14
                This directory contains QLoRA adapter weights for a Gemma 3 model fine-tuned on philosophical texts.

                ## Loading the model

                ```python
                from transformers import Gemma3ForCausalLM, AutoTokenizer
                from peft import PeftModel

                # Load base model
                base_model = Gemma3ForCausalLM.from_pretrained("google/gemma-3-1b-it", device_map="auto", trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", trust_remote_code=True)

                # Load adapter
                model = PeftModel.from_pretrained(base_model, "models/aritstotlegemma")
            