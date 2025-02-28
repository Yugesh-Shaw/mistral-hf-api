import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
import requests
from io import BytesIO

# Set Hugging Face cache to a writable directory (optional)
os.environ['HF_HOME'] = '/tmp/huggingface_cache'  # Customize cache directory if necessary

# Initialize FastAPI app
app = FastAPI()

# Define Hugging Face model URL (replace with your desired model)
MODEL_URL = "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/pytorch_model.bin"
TOKENIZER_URL = "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/tokenizer.json"

# Function to download model and tokenizer from Hugging Face URL
def download_model_and_tokenizer(model_url, tokenizer_url):
    # Download model
    model_response = requests.get(model_url)
    model = AutoModelForCausalLM.from_pretrained(BytesIO(model_response.content))

    # Download tokenizer
    tokenizer_response = requests.get(tokenizer_url)
    tokenizer = AutoTokenizer.from_pretrained(BytesIO(tokenizer_response.content))
    
    return model, tokenizer

# Load model and tokenizer dynamically from Hugging Face URL
model, tokenizer = download_model_and_tokenizer(MODEL_URL, TOKENIZER_URL)

# Ensure model runs on GPU if available, otherwise fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

@app.post("/paraphrase")
async def paraphrase(request: dict):
    text = request["text"]
    prompt = f"Paraphrase this: {text}"
    
    # Tokenize the input text
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate paraphrased text
    outputs = model.generate(**inputs, max_new_tokens=200)
    
    # Decode the generated output
    paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"paraphrased_text": paraphrased_text}
