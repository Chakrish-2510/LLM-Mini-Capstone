# LLM Mini-Capstone: From Prompt to Evaluation

## Project objective
Demonstrate how a pretrained Large Language Model (LLM) receives a prompt, tokenizes it, generates a response, and is evaluated using a small test set. This is an inference-and-evaluation project; it does **not** train an LLM from scratch.

## Technology stack
- Python — application logic
- Hugging Face Transformers — tokenizer and pretrained model interface
- `google/flan-t5-small` — pretrained instruction-tuned text-to-text model
- PyTorch — model inference
- Streamlit — interactive web interface
- Matplotlib — evaluation graphs
- CSV — test prompts and recorded results
- Google Colab or a local Python environment — execution

## Pipeline
1. **Input:** prompt/instruction from the user or test CSV.
2. **Tokenization:** converts text into token IDs.
3. **Model processing:** the pretrained Transformer generates output tokens.
4. **Decoding:** token IDs are converted back into readable text.
5. **Outputs:** generated response, response time, input/output token counts.
6. **Evaluation:** run a fixed prompt set; compute keyword coverage as a simple relevance proxy; record human ratings and notes.
7. **Analysis:** save CSV results and graphs for response time and keyword coverage.

## Run locally
Use Python 3.10 or 3.11.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the pretrained model and requires internet access. CPU inference works but may be slower.

## Run the benchmark
```bash
python evaluate.py
```

Generated files are written to `outputs/`:
- `evaluation_results.csv`
- `response_time.png`
- `keyword_coverage.png`
- `summary.txt`

Open the CSV and complete `human_rating_1_to_5` and `review_notes` after manually reviewing each response.

## Run in Google Colab
1. Upload/extract this project folder into Colab, or clone your GitHub repository.
2. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
3. Run the benchmark:
   ```python
   !python evaluate.py
   ```
4. To use the interactive interface, run Streamlit in a suitable local environment or configure a Colab tunnel according to your institution's rules.

## Evaluation methodology
- **Response time:** elapsed seconds for generation (model loading is excluded).
- **Keyword coverage:** percentage of expected keywords found in the generated response. This is only a rough proxy and can penalize valid paraphrases.
- **Human review:** rate each answer from 1–5 for correctness, relevance, clarity, and completeness. Add notes for factual errors or hallucinations.
- **Fair comparison:** keep the same prompts, settings, and hardware when comparing runs. Record the model name and date.

## Limitations and responsible use
- Results depend on model version, hardware, prompt wording, and generation settings.
- Keyword overlap does not establish factual accuracy.
- A small test set is not enough to make broad claims about all LLMs.
- The model may produce incorrect or incomplete answers. Verify important claims.
- Do not upload private or sensitive data to an external model or repository.

## Suggested viva explanation
“We implemented an end-to-end LLM inference and evaluation pipeline. A prompt is tokenized, processed by a pretrained Transformer model, and decoded into a response. We measure generation latency and token counts, compare outputs against expected keywords, and include a human-review column for quality assessment. The project demonstrates methodology and evaluation rather than simply prompting an online AI tool.”
