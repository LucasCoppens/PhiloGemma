# PhiloGemma: Aristotelian Wisdom in a Modern LLM

PhiloGemma is a philosophical fine-tuning of Google's Gemma 3 language model, trained on Aristotle's works to embody the reasoning patterns and linguistic style of classical philosophy. However, the code in this package can be adapted to create answers finetuned to the style of any input dataset.

The model implements a lightweight QLoRA adapter (~3M parameters) on the Gemma3 Q_proj, K_proj, V_proj and O_proj modules. 

Scripts are included for both training and inference. In the current implementation, training data is expected in the .txt format. 

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/philogemma.git
cd philogemma

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Interactive CLI

```bash
# Use base Gemma 3 model
python scripts/inference.py

# Use PhiloGemma (fine-tuned model)
python scripts/inference.py --model_path models/aristotlegemma
python scripts/inference.py --model_path models/aristotlegemma --adapter_scale 1.4 # above 1 increases influence of adapter contribution
python scripts/inference.py --model_path models/aristotlegemma --temperature 1.5 # for more varied and experimental results
```

### Training Your Own Model

```bash
# Fine-tune on your own philosophical texts
python scripts/train.py --input_dir data/your_texts --output_dir models/your_model
```

### Using the Model in Code

```python
from src.model import PhiloGemmaModel

# Load model
model, tokenizer = PhiloGemmaModel.load_for_inference(
    adapter_path="models/aristotlegemma",  # Use None for base model
    base_model_name="google/gemma-3-1b-it"
)

# Generate response
response = PhiloGemmaModel.generate_response(
    model=model,
    tokenizer=tokenizer,
    prompt="What is the nature of virtue?",
    temperature=0.8
)

print(response)
```

## Hugging Face Integration

The PhiloGemma adapter is available on Hugging Face:

```python
from transformers import Gemma3ForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = Gemma3ForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    device_map="auto", 
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-3-1b-it", 
    trust_remote_code=True
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "LCoppens/aristotlegemma")
```

## Project Structure

```
philogemma/
├── data/                  # Training data directory
├── logs/                  # Training and inference logs
├── models/                # Model weights and cache
│   ├── cache/             # Base model cache
│   └── philogemma/        # Fine-tuned adapter weights
├── scripts/               # Command-line scripts
│   ├── inference.py       # Interactive CLI
│   └── train.py           # Training script
├── src/                   # Source code
│   ├── dataloader.py      # Data loading utilities
│   ├── model.py           # Model definition and utilities
│   └── trainer.py         # Training utilities
└── README.md              # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- PEFT (Parameter-Efficient Fine-Tuning)
- BitsAndBytes (for quantization)

## Examples

### Knowledge vs. Wisdom

**User:** How can we differentiate between knowledge and wisdom?

**Standard Gemma 3:**
```
Distinguishing between knowledge and wisdom is a complex process, but it often boils down to application. Knowledge is simply information – facts and data you've learned. Wisdom, however, involves using that knowledge ethically and reflecting on its implications. It's about understanding *why* something is true and how it applies to different situations. True wisdom requires experience and reflection, not just accumulation of facts. It's about recognizing patterns and making informed decisions. Often, knowledge fades without context; wisdom integrates it into a broader perspective. Ultimately, wisdom considers both knowing *and* being able to act thoughtfully.
```

**PhiloGemma:**
```
We distinguish between knowledge and wisdom by considering both their methods of attaining truth: knowledge is attained through reasoned inquiry, while wisdom derives from experience and reflection, especially in relation to fundamental principles. Knowledge is essentially the product of reason's power of combining particulars together to form generalities, as is shown in mathematics and geometry. For it involves bringing facts together to construct truths. Wisdom, however, extends beyond merely acquiring facts, but also involves the understanding of the underlying principles or causes, which will always be universal and applicable, as is evident in many things: for instance, the mathematical principle that 2 + 3 = 5 is the same for all numbers. Whereas, knowledge may be grasped through observation alone, though this is not the case of any particular science, wisdom requires one to know what principles exist and to know why they exist; for when we have an understanding of these principles, the fact that two numbers are 10 greater than five becomes a matter of fact only through the application of the 2+3 = 5 principle. Therefore, if someone attains knowledge by simply observing, he does not yet possess wisdom; whereas if he has knowledge of some principles, he has already come to possess wisdom. Thus, the former has a method of obtaining truth, whilst the latter has knowledge of the nature of the things in question. The method of knowing, therefore, consists of observing and studying, of examining, of searching, of looking into details and seeing what is hidden, while the method of wisdom is of observation, as also of thinking and reflecting, and of inquiring into the causes.
```

**User:** Tell me about the nature of sunlight

**Standard Gemma 3:**
```
Sunlight is a form of electromagnetic radiation, primarily visible light. It’s produced by the nuclear fusion reactions within the Sun’s core. This energy travels outwards as waves, creating what we perceive as illumination. Sunlight contains different wavelengths – shorter wavelengths like blue and violet appear to bounce off surfaces more readily, while longer wavelengths like red appear less so.  It’s crucial for photosynthesis in plants, fueling the food chain. Without solar energy, life as we know it wouldn't exist. The intensity and color depend on factors like cloud cover and atmospheric conditions. 
```

**PhiloGemma:**
```
Sunlight is the primary source of energy for all living things. It is a radiant energy, a form of light that carries heat and warmth. It comes from celestial bodies, mainly the sun. From these celestial bodies, the Sun emits rays which fall to earth as light. A great part of this light makes up for the bulk of our body’s energy. But light itself can be used as a catalyst in chemical reactions. The light produced by the sun is the cause of this; it causes growth and development. Moreover, the light that comes from the sun has such characteristics as to provide many things to us—for instance the plants we eat and the animals, the very beings that we employ as instruments for work and pleasure.
```