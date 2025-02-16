from transformers import PretrainedConfig, PreTrainedModel, AutoModelForCausalLM, AutoConfig
from smollm135_model import SmoLM135_Model
import torch

class SmoLM135Config(PretrainedConfig):
    model_type = "smolm135"
    
    def __init__(
        self,
        hidden_size=576,
        vocab_size=49152,
        num_hidden_layers=30,
        num_attention_heads=9,
        intermediate_size=1536,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

class SmoLM135ForCausalLM(PreTrainedModel):
    config_class = SmoLM135Config
    supports_gradient_checkpointing = True
    base_model_prefix = "smolm135"
    
    def __init__(self, config):
        super().__init__(config)
        model_dict = {
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "intermediate_size": config.intermediate_size,
            "max_position_embeddings": config.max_position_embeddings,
            "rms_norm_eps": config.rms_norm_eps,
            "tie_word_embeddings": config.tie_word_embeddings
        }
        self.model = SmoLM135_Model(model_dict)
        
        if config.tie_word_embeddings:
            self.model.lm_head.weight = self.model.token_embeddings.weight
            
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids)
    
    def generate(self, input_ids, max_length=100, temperature=0.7, **kwargs):
        device = next(self.parameters()).device
        context = input_ids.to(device)
        
        for _ in range(max_length - input_ids.size(1)):
            with torch.no_grad():
                logits, _ = self.model(context)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                context = torch.cat([context, next_token], dim=1)
                
        return context

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

# Register the model with Auto classes
AutoConfig.register(SmoLM135Config.model_type, SmoLM135Config)
AutoModelForCausalLM.register(SmoLM135Config, SmoLM135ForCausalLM) 