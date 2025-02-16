import os
import math
import time
import torch
import torch.nn.functional as F
import tiktoken
import logging
from datetime import datetime
from smollm135_model import SmoLM135_Model, config

# Setup logging
def setup_logging():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'continued_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Configure logging
    logger = logging.getLogger('continued_training')
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
    
    return logger

def log_info(msg):
    print(msg)  # Print to terminal
    logger.info(msg)  # Log to file

def log_warning(msg):
    print(f"WARNING: {msg}")  # Print to terminal
    logger.warning(msg)  # Log to file

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')

# Training parameters
additional_steps = 50
max_lr = 6e-4
min_lr = max_lr * 0.1
grad_clip = 1.0

# DataLoader class
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

def generate_sample(model, tokenizer, start_text="The ", max_new_tokens=50):
    model.eval()
    context = torch.tensor(tokenizer.encode(start_text), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            for _ in range(max_new_tokens):
                logits, _ = model(context)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                context = torch.cat([context, next_token], dim=1)
    
    decoded = tokenizer.decode(context[0].tolist())
    model.train()
    return decoded

def continue_training():
    global logger
    logger = setup_logging()
    log_info(f"Continuing training on device: {device}")
    
    # Initialize tokenizer
    global enc
    enc = tiktoken.get_encoding('gpt2')
    
    # Load model and checkpoint
    model = SmoLM135_Model(config)
    model.to(device)
    
    # Find the most recent final model checkpoint in models directory
    models_dir = 'models'
    if not os.path.exists(models_dir):
        raise ValueError("Models directory not found!")
        
    checkpoints = [f for f in os.listdir(models_dir) if f.startswith('final_model_') and f.endswith('.pt')]
    if not checkpoints:
        raise ValueError("No final model checkpoint found in models directory!")
    
    latest_checkpoint = max(checkpoints)
    checkpoint_path = os.path.join(models_dir, latest_checkpoint)
    log_info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize optimizer
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=max_lr,
        device_type=device
    )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Initialize data loader
    train_loader = DataLoaderLite(B=32, T=128, vocab_size=model.vocab_size)
    
    start_step = checkpoint['step']
    log_info(f"Continuing from step {start_step} for {additional_steps} more steps")
    
    # Training loop
    for step in range(additional_steps):
        t0 = time.time()
        
        try:
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
                
            if not torch.isfinite(loss):
                log_warning(f"Non-finite loss detected: {loss.item()}")
                continue
                
            loss.backward()
            
            if grad_clip != 0.0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
        except RuntimeError as e:
            log_warning(f"Error during training: {str(e)}")
            continue
            
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = train_loader.B * train_loader.T / dt
        
        if step % 5 == 0:  # More frequent logging for short continuation
            log_info(f"step {start_step + step} | loss {loss.item():.4f} | {tokens_per_sec:.0f} tokens/sec")
            
            # Generate sample every 10 steps
            if step % 10 == 0:
                log_info("\nGenerating sample:")
                sample_text = generate_sample(model, enc)
                log_info(sample_text)
                log_info("\n")
    
    # Save final continued model
    continued_dir = 'continued_models'
    os.makedirs(continued_dir, exist_ok=True)
    final_continued_path = os.path.join(continued_dir, f'continued_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
    log_info(f"Saving continued model to {final_continued_path}")
    torch.save({
        'step': start_step + additional_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, final_continued_path)
    
    log_info(f"Continued training completed. Final loss: {loss.item():.4f}")

if __name__ == "__main__":
    continue_training() 