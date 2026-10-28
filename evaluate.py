"""Run a small, transparent prompt benchmark and create CSV/PNG results.

The reference answers and keyword lists are illustrative evaluation examples.
Keyword overlap is a simple proxy, NOT a complete measure of correctness.
Review the outputs manually before presenting results.
"""
import csv
import time
from pathlib import Path
import matplotlib.pyplot as plt
from llm_core import load_model, generate_answer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "test_prompts.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def keyword_overlap(answer, keywords):
    terms = [x.strip().lower() for x in keywords.split("|") if x.strip()]
    if not terms:
        return 0.0
    answer_lower = answer.lower()
    return round(sum(term in answer_lower for term in terms) / len(terms) * 100, 2)

def main():
    tokenizer, model = load_model()
    with DATA.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    print(f"Evaluating {len(rows)} prompts with { 'google/flan-t5-small' } ...")
    for i, row in enumerate(rows, start=1):
        answer, seconds, input_tokens, output_tokens = generate_answer(
            row["prompt"], tokenizer, model, max_new_tokens=100
        )
        overlap = keyword_overlap(answer, row["keywords"])
        result = {
            "id": row["id"],
            "category": row["category"],
            "prompt": row["prompt"],
            "reference_answer": row["reference_answer"],
            "model_output": answer,
            "response_time_seconds": round(seconds, 3),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "keyword_coverage_percent": overlap,
            "human_rating_1_to_5": "",
            "review_notes": "",
        }
        results.append(result)
        print(f"{i}/{len(rows)} | {seconds:.2f}s | keyword coverage {overlap}%")

    csv_path = OUT / "evaluation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Plot response time per prompt
    plt.figure(figsize=(10, 5))
    plt.bar([r["id"] for r in results], [r["response_time_seconds"] for r in results])
    plt.title("LLM Response Time by Test Prompt")
    plt.xlabel("Prompt ID")
    plt.ylabel("Generation time (seconds)")
    plt.tight_layout()
    plt.savefig(OUT / "response_time.png", dpi=160)
    plt.close()

    # Plot simple keyword coverage
    plt.figure(figsize=(10, 5))
    plt.bar([r["id"] for r in results], [r["keyword_coverage_percent"] for r in results])
    plt.title("Keyword Coverage (Simple Proxy for Relevance)")
    plt.xlabel("Prompt ID")
    plt.ylabel("Keyword coverage (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(OUT / "keyword_coverage.png", dpi=160)
    plt.close()

    avg_time = sum(r["response_time_seconds"] for r in results) / len(results)
    avg_coverage = sum(r["keyword_coverage_percent"] for r in results) / len(results)
    summary_path = OUT / "summary.txt"
    summary_path.write_text(
        f"Model: google/flan-t5-small\n"
        f"Prompts evaluated: {len(results)}\n"
        f"Average generation time: {avg_time:.3f} seconds\n"
        f"Average keyword coverage: {avg_coverage:.2f}%\n\n"
        "Important: keyword coverage is a weak automatic proxy. It does not prove factual correctness.\n"
        "Review each output, fill human_rating_1_to_5 and review_notes in evaluation_results.csv,\n"
        "and discuss factuality, relevance, clarity, and limitations in your report.\n",
        encoding="utf-8",
    )
    print(f"\nDone. Results saved in: {OUT}")
    print(f"Average generation time: {avg_time:.3f} seconds")
    print(f"Average keyword coverage: {avg_coverage:.2f}%")
    print("Please manually review outputs and complete the human ratings.")

if __name__ == "__main__":
    main()
