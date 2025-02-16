import gradio as gr
import torch
import tiktoken
from modeling_smolm135 import SmoLM135ForCausalLM, SmoLM135Config

# Load model and tokenizer
model_id = "debasisha/smolm135"
config = SmoLM135Config.from_pretrained(model_id)
model = SmoLM135ForCausalLM.from_pretrained(model_id)
enc = tiktoken.get_encoding('gpt2')

def generate_text(prompt, max_length=100, temperature=0.7):
    # Encode the prompt
    input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            pad_token_id=50256,
            eos_token_id=50256
        )
    
    # Decode and return
    generated_text = enc.decode(output_ids[0].tolist())
    return generated_text

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
        gr.Slider(minimum=10, maximum=200, value=100, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="SmoLM135 Text Generator",
    description="Enter a prompt and the model will continue generating text.",
    examples=[
        ["The quick brown fox", 100, 0.7],
        ["Once upon a time", 150, 0.8],
        ["In a galaxy far far away", 120, 0.6]
    ]
)

# Launch the app
iface.launch() 