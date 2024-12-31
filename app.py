from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


app = FastAPI()


MODEL_PATH = "transformer_final.pth"
tokenizer = AutoTokenizer.from_pretrained("gpt2")  
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)


class Query(BaseModel):
    input_text: str

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model.eval()
    print("Model and tokenizer are ready!")

@app.post("/generate/")
async def generate_text(query: Query):
    input_text = query.input_text
    try:
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        output_ids = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {"input": input_text, "output": output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "Model API is up and running!"}
