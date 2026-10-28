"""Shared model loading and generation functions."""
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

def generate_answer(prompt, tokenizer, model, max_new_tokens=96):
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_tokens = int(encoded["input_ids"].shape[1])
    start = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            num_beams=2,
        )
    elapsed = time.perf_counter() - start
    answer = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    output_tokens = int(generated.shape[1])
    return answer, elapsed, input_tokens, output_tokens
