import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect
import logging

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config['hidden_size'] % config['num_attention_heads'] == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config['hidden_size'], 3 * config['hidden_size'])
        # output projection
        self.c_proj = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.c_proj.NANGPT_SCALE_INIT = 1
        # regularization
        self.num_attention_heads = config['num_attention_heads']
        self.hidden_size = config['hidden_size']
        self.register_buffer("bias", torch.tril(torch.ones(config['max_position_embeddings'], config['max_position_embeddings'])).view(1, 1, config['max_position_embeddings'], config['max_position_embeddings']))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (hidden_size)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), num_attention_heads=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.hidden_size, dim=2)
        k = k.view(B, T, self.num_attention_heads, C // self.num_attention_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_attention_heads, C // self.num_attention_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_attention_heads, C // self.num_attention_heads).transpose(1, 2) # (B, nh, T, hs)

        # Use Flash Attention if available, otherwise fall back to regular attention
        if hasattr(F, 'scaled_dot_product_attention') and torch.cuda.is_available():
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class SmoLM135_Model(nn.Module):
    def __init__(self, config):
        super(SmoLM135_Model, self).__init__()
        
        # Load model configuration from YAML
        self.hidden_size = config['hidden_size']
        self.vocab_size = config['vocab_size']
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_attention_heads = config['num_attention_heads']
        self.intermediate_size = config['intermediate_size']
        self.max_position_embeddings = config['max_position_embeddings']
        self.rms_norm_eps = config['rms_norm_eps']
        self.tie_word_embeddings = config['tie_word_embeddings']

        # Embedding layers
        self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config,self.hidden_size, self.num_attention_heads, self.intermediate_size, self.rms_norm_eps)
            for _ in range(self.num_hidden_layers)
        ])

        # Output layer
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)
        if self.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, input_ids, targets=None):
        # Add input validation
        if torch.max(input_ids) >= self.vocab_size:
            raise ValueError(f"Input ids contain token(s) outside of vocabulary range. "
                            f"Max token ID: {torch.max(input_ids).item()}, "
                            f"Vocab size: {self.vocab_size}")
        
        # Input embeddings
        seq_length = input_ids.size(1)
        if seq_length > self.max_position_embeddings:
            raise ValueError(f"Input sequence length ({seq_length}) exceeds maximum "
                            f"position embeddings ({self.max_position_embeddings})")
        
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        x = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)

        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final linear layer
        logits = self.lm_head(x)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            if torch.max(targets) >= self.vocab_size:
                raise ValueError(f"Target ids contain token(s) outside of vocabulary range. "
                               f"Max token ID: {torch.max(targets).item()}, "
                               f"Vocab size: {self.vocab_size}")
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        """
        Configure the optimizer with weight decay and learning rate.
        """
        # Start with all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Create optim groups. Weight decay for 2D parameters, no decay for others
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        msg = f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        print(msg)
        if hasattr(logging.getLogger(), 'info'):
            logging.getLogger().info(msg)
        
        msg = f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        print(msg)
        if hasattr(logging.getLogger(), 'info'):
            logging.getLogger().info(msg)
        
        # Create AdamW optimizer and use the fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        
        msg = f"using fused AdamW: {use_fused}"
        print(msg)
        if hasattr(logging.getLogger(), 'info'):
            logging.getLogger().info(msg)
        
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )
        
        return optimizer

class TransformerLayer(nn.Module):
    def __init__(self, config,hidden_size, num_attention_heads, intermediate_size, rms_norm_eps):
        super(TransformerLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.head_dim = hidden_size // num_attention_heads

        assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"

        # Self-attention
        self.self_attn = CausalSelfAttention(config)
        self.ln_1 = nn.LayerNorm(hidden_size)

        # Feedforward layers
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = F.silu  # Activation function (SiLU)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.linear2.NANOGPT_SCALE_INIT = 1

        # Normalization
        self.norm1 = nn.LayerNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, x):
        # Self-attention block
        attn_output = self.self_attn(self.ln_1(x))
        x = x + attn_output  # Residual connection
        x = self.norm1(x)

        # Feedforward block
        ff_output = self.linear2(self.activation(self.linear1(x)))
        x = x + ff_output  # Residual connection

        return x

# Load model configuration from YAML-like dictionary (parsed from YAML)
config = {
    "hidden_size": 576,
    "vocab_size": 50257,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "intermediate_size": 1536,
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": True
}