import os
import torch
from smollm135_model import config as model_config
from modeling_smolm135 import SmoLM135Config, SmoLM135ForCausalLM
from huggingface_hub import HfApi, create_repo

def convert_to_hf_model():
    # Load your trained model
    model_path = "models/final_model_20250216_125219.pt"
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create HF config
    config = SmoLM135Config(
        hidden_size=model_config["hidden_size"],
        vocab_size=model_config["vocab_size"],
        num_hidden_layers=model_config["num_hidden_layers"],
        num_attention_heads=model_config["num_attention_heads"],
        intermediate_size=model_config["intermediate_size"],
        max_position_embeddings=model_config["max_position_embeddings"],
        rms_norm_eps=model_config["rms_norm_eps"],
        tie_word_embeddings=model_config["tie_word_embeddings"]
    )
    
    # Initialize HF model
    hf_model = SmoLM135ForCausalLM(config)
    hf_model.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save model and config
    output_dir = "hf_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config.save_pretrained(output_dir)
    
    # Save model with safe serialization disabled
    hf_model.save_pretrained(output_dir, safe_serialization=False)
    
    # Push to Hub
    api = HfApi()
    create_repo("debasisha/smolm135", private=False)
    api.upload_folder(
        folder_path=output_dir,
        repo_id="debasisha/smolm135",
        repo_type="model"
    )

if __name__ == "__main__":
    convert_to_hf_model() 