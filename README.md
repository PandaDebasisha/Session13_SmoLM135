# SmoLM135: A Lightweight Language Model

SmoLM135 is a compact transformer-based language model designed for efficient text generation. It features a balanced architecture optimized for performance while maintaining a reasonable parameter count.

## Model Architecture

- **Base Architecture**: Transformer
- **Parameters**:
  - Hidden Size: 576
  - Vocabulary Size: 49,152 (GPT-2 tokenizer compatible)
  - Number of Layers: 30
  - Attention Heads: 9
  - Intermediate Size: 1,536
  - Max Position Embeddings: 2,048
  - Total Parameters: ~135M
**Parameter count breakdown:**
- Token Embeddings: 49,152 × 576 = 28,311,552
- Self-Attention (per layer):
  - Q/K/V Projections: 3 × (576 × 576) = 995,328
  - Output Projection: 576 × 576 = 331,776
- MLP (per layer):
  - Gate Projection: 576 × 1,536 = 884,736
  - Up Projection: 576 × 1,536 = 884,736
  - Down Projection: 1,536 × 576 = 884,736
- Layer Norm (per layer): 576 × 2 = 1,152
- Total- 135M
## Features

- Efficient transformer implementation with flash attention support
- Weight tying between input embeddings and output layer
- Mixed precision training support
- Checkpoint saving and resumption
- Gradient clipping and learning rate scheduling
- Automatic mixed precision (AMP) training

## Installation

## Project Structure
project/
├── models/ # Saved model checkpoints
│ └── final_model_.pt
├── continued_models/ # Continued training checkpoints
│ └── continued_model_.pt
├── logs/ # Training logs
│ ├── training_.log
│ └── continued_training_.log
├── input.txt # Training data
├── Model_smollm135_trainer.py # Main training script
├── continue_training.py # Script for continued training
├── modeling_smolm135.py # HuggingFace model implementation
├── smollm135_model.py # Core model architecture
├── convert_to_hf.py # Script to convert model to HF format
└── requirements.txt # Dependencies

## Training Steps

1. **Prepare Training Data**
   ```bash
   # Place your training text in input.txt
   # The model uses tiktoken GPT-2 tokenizer
   ```

2. **Initial Training**
   ```bash
   python Model_smollm135_trainer.py
   ```
   - Training Parameters:
     - Batch Size: 32
     - Sequence Length: 128
     - Learning Rate: 6e-4 with cosine decay
     - Warmup Steps: 2,000
     - Total Steps: 5,000
     - Checkpoint Frequency: 500 steps
     - Gradient Clipping: 1.0

3. **Continue Training** (Optional)
   ```bash
   python continue_training.py
   ```
   - Continues for additional 50 steps
   - Uses the most recent final model checkpoint
   - Maintains same batch size and sequence length
   - Saves to separate continued_models directory

4. **Convert to HuggingFace Format**
   ```bash
   python convert_to_hf.py
   ```
   - Converts the trained model to HuggingFace format
   - Pushes to HuggingFace Hub
   - Creates model card and configuration

## Model Usage

### Local Usage
python
import torch
from smollm135_model import SmoLM135_Model, config
import tiktoken
Initialize model
model = SmoLM135_Model(config)
model.load_state_dict(torch.load("models/final_model_TIMESTAMP.pt")['model_state_dict'])
model.eval()
Initialize tokenizer
enc = tiktoken.get_encoding('gpt2')
Generate text
def generate_text(prompt, max_length=100, temperature=0.7):
input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0)
with torch.no_grad():
output_ids = model.generate(
input_ids,
max_length=max_length,
temperature=temperature
)
return enc.decode(output_ids[0].tolist())

### HuggingFace Hub Usage

```python
from transformers import AutoModelForCausalLM, AutoConfig
import tiktoken

# Load model from HuggingFace
model_id = "debasisha/smolm135"
model = AutoModelForCausalLM.from_pretrained(model_id)
enc = tiktoken.get_encoding('gpt2')

# Generate text
# (same as above)
```

## Training Monitoring

- **Logs**: Check the `logs` directory for detailed training logs
- **Checkpoints**: 
  - Regular checkpoints in `checkpoints/latest.pt`
  - Final model in `models/final_model_TIMESTAMP.pt`
  - Continued training in `continued_models/`
- **Sample Generation**: Model generates sample text every 500 steps

## Technical Details

### Model Components
1. **Attention Mechanism**
   - Multi-head attention with 9 heads
   - Flash Attention when available
   - Causal (left-to-right) attention mask

2. **Position Embeddings**
   - Learned position embeddings
   - Maximum sequence length: 2048 tokens

3. **Optimization**
   - AdamW optimizer
   - Weight decay: 0.1
   - Learning rate scheduling with warmup
   - Gradient clipping at 1.0
   - Mixed precision training (bfloat16)

4. **Memory Efficiency**
   - Weight tying between embedding and output layer
   - Efficient attention implementation
   - Gradient checkpointing support

## Requirements

```txt
torch>=2.0.0
transformers>=4.30.0
tiktoken
gradio>=3.35.2
huggingface_hub
```

## HuggingFace Integration

- Model: [debasisha/smolm135](https://huggingface.co/debasisha/smolm135)
- Demo Space: [debasisha/smolm135-demo](https://huggingface.co/spaces/debasisha/smolm-demo)

## Troubleshooting

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Clear cache: `torch.cuda.empty_cache()`

2. **Training Resumption**
   - Checkpoints auto-save every 500 steps
   - Automatically resumes from last checkpoint
   - Manual resume possible using continue_training.py

3. **Common Issues**
   - Token validation errors: Check vocab_size matches tokenizer
   - NaN loss: Reduce learning rate or check for data issues
   - CUDA errors: Update PyTorch or check GPU compatibility

## License

[Free Free Free]

## Citation

```bibtex
@software{smolm135,
  author = {Debasisha panda},
  title = {SmoLM135: A Lightweight Language Model},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/your-username/smolm135}
}
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- Thanks to the Hugging Face team for their transformers library
- [Any other acknowledgments]

