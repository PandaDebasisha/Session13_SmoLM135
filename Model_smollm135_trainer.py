import os
import math
import time
import torch
import torch.nn.functional as F
import tiktoken
from smollm135_model import SmoLM135_Model, config
import logging
from datetime import datetime

# Set up logging first
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Configure logging
logger = logging.getLogger('training')
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console_handler)

def log_info(msg):
    print(msg)  # Print to terminal
    logger.info(msg)  # Log to file

def log_warning(msg):
    print(f"WARNING: {msg}")  # Print to terminal
    logger.warning(msg)  # Log to file

# Device configuration
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
log_info(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Training hyperparameters
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 2000
max_steps = 5000
checkpoint_every = 500
grad_clip = 1.0

# Enable tensor cores for faster training
torch.set_float32_matmul_precision('high')

# Initialize tokenizer
enc = tiktoken.get_encoding('gpt2')

class DataLoaderLite:
    def __init__(self, B, T, vocab_size):
        self.B = B
        self.T = T
        self.vocab_size = vocab_size

        # Load tokens from disk
        with open('input.txt', 'r') as f:
            text = f.read()
        tokens = enc.encode(text)
        
        # Filter out tokens that are outside the vocabulary range
        tokens = [t if t < vocab_size else vocab_size-1 for t in tokens]
        log_info(f"Max token ID: {max(tokens)}")
        
        self.tokens = torch.tensor(tokens)
        log_info(f'Loaded {len(self.tokens)} tokens')
        log_info(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
            
        return x, y

def get_lr(it):
    # Learning rate schedule
    if it < warmup_steps:
        return max(min_lr, max_lr * (it + 1) / warmup_steps)
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    decay_ratio = max(0.0, min(1.0, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return max(min_lr, min_lr + coeff * (max_lr - min_lr))

def generate_sample(model, start_text="The ", max_new_tokens=50):
    model.eval()
    context = torch.tensor(enc.encode(start_text), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            for _ in range(max_new_tokens):
                logits, _ = model(context)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                context = torch.cat([context, next_token], dim=1)
    
    decoded = enc.decode(context[0].tolist())
    model.train()
    return decoded

def save_checkpoint(path, step, model, optimizer, loss):
    # Get directory path if it exists, otherwise use current directory
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:  # If there's a directory path
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    temp_path = f"{path}.tmp"
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, path)
        log_info(f"Checkpoint saved successfully to {path}")
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        log_warning(f"Error saving checkpoint: {str(e)}")
        raise e

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    # Initialize model
    model = SmoLM135_Model(config)
    model.to(device)
    
    # Initialize optimizer
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=max_lr,
        device_type=device
    )
    
    # Initialize data loader
    train_loader = DataLoaderLite(B=32, T=128, vocab_size=model.vocab_size)
    
    # Create checkpoint directory
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load checkpoint if exists
    start_step = 0
    if os.path.exists(os.path.join(checkpoint_dir, 'latest.pt')):
        log_info("Loading checkpoint...")
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'latest.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        log_info(f"Resuming from step {start_step}")
    
    # Training loop
    for step in range(start_step, max_steps):
        t0 = time.time()
        
        try:
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
                
            if not torch.isfinite(loss):
                log_warning(f"WARNING: Non-finite loss detected: {loss.item()}")
                continue
                
            loss.backward()
            
            if grad_clip != 0.0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Update learning rate
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            optimizer.step()
            
        except RuntimeError as e:
            if "device-side assert triggered" in str(e):
                log_warning(f"WARNING: CUDA assertion failed. Input shape: {x.shape}, "
                              f"Max token ID: {torch.max(x).item()}, "
                              f"Vocab size: {model.vocab_size}")
                continue
            elif "out of memory" in str(e):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                log_warning("WARNING: out of memory, skipping batch")
                continue
            raise e
            
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = train_loader.B * train_loader.T / dt
        
        if step % 10 == 0:
            log_info(f"step {step} | loss {loss.item():.4f} | lr {lr:.6f} | {tokens_per_sec:.0f} tokens/sec")
        
        # Save checkpoint and generate sample
        if step > 0 and step % checkpoint_every == 0:
            log_info("\nSaving checkpoint...")
            save_checkpoint(
                os.path.join(checkpoint_dir, 'latest.pt'),
                step,
                model,
                optimizer,
                loss.item()
            )
            
            log_info("\nGenerating sample:")
            sample_text = generate_sample(model)
            log_info(sample_text)
            log_info("\n")
    
    # Create models directory for final checkpoints
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save final model
    final_model_path = os.path.join(models_dir, f'final_model_{get_timestamp()}.pt')
    log_info(f"Saving final model to {final_model_path}...")
    save_checkpoint(final_model_path, max_steps, model, optimizer, loss.item())
    log_info(f"Training completed. Final loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
