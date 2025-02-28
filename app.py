from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
import torch

app = FastAPI()

# ✅ Load Mistral 7B Model
model_name = "mistralai/Mistral-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

@app.post("/paraphrase")
async def paraphrase(request: dict):
    text = request["text"]
    prompt = f"Paraphrase this: {text}"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"paraphrased_text": response}
