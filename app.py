"""LLM Mini-Capstone: interactive demo using a pretrained instruction-tuned model."""
import time
import streamlit as st
from llm_core import load_model, generate_answer

st.set_page_config(page_title="LLM Mini-Capstone", page_icon="🧠", layout="centered")
st.title("🧠 Large Language Model Mini-Capstone")
st.caption("Input → Tokenization → Pretrained LLM → Generated response → Evaluation")

with st.sidebar:
    st.header("Model")
    st.write("Model: google/flan-t5-small")
    st.write("Framework: Hugging Face Transformers + PyTorch")
    st.info("The model is pretrained. This project demonstrates inference and evaluation, not training an LLM from scratch.")

@st.cache_resource
def get_model():
    return load_model()

prompt = st.text_area(
    "Enter a question or instruction",
    value="Explain how solar energy works in simple words.",
    height=120,
)
max_tokens = st.slider("Maximum new tokens", min_value=32, max_value=160, value=96, step=16)

if st.button("Generate response", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Loading/running the model..."):
            tokenizer, model = get_model()
            answer, elapsed, input_tokens, output_tokens = generate_answer(
                prompt, tokenizer, model, max_new_tokens=max_tokens
            )
        st.subheader("Model output")
        st.write(answer)
        st.subheader("Run metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Response time", f"{elapsed:.2f} s")
        c2.metric("Input tokens", str(input_tokens))
        c3.metric("Output tokens", str(output_tokens))
        st.caption("These metrics describe this run and can vary by hardware, model loading, and prompt.")
