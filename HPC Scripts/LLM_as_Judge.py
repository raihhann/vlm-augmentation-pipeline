import concurrent.futures
import os
import re
import numpy as np
import ollama
import pandas as pd
from tqdm import tqdm

# ==========================================
# 1. Configuration Settings
# ==========================================
MODEL_NAME = "deepseek-r1:32b"
INPUT_FILE = "INPUT CSV FILE PATH HERE" 
OUTPUT_FILE = "semantic_correctness.csv"
RUNS_PER_EVAL = 3   # 3 evaluation passes per row
NUM_WORKERS = 4     # Number of parallel worker threads (optimal for Dual RTX 4090)
SAVE_INTERVAL = 10  # Auto-save progress every 10 rows


# ==========================================
# 2. DeepSeek Output Parser
# ==========================================
def extract_score_from_deepseek(raw_text: str) -> float:
    """Strips <think>...</think> tags and extracts the final integer score."""
    # 1. Remove complete thinking blocks
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    
    # 2. Handle unclosed <think> tags if generation was cut short
    if "<think>" in cleaned:
        cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()
    
    # 3. Extract numbers from the cleaned text
    numbers = re.findall(r"\d+", cleaned)
    if numbers:
        score = float(numbers[-1])
        return min(max(score, 0.0), 100.0)
        
    # Fallback: search entire raw response if no numbers were found outside <think>
    raw_numbers = re.findall(r"\d+", raw_text)
    if raw_numbers:
        score = float(raw_numbers[-1])
        return min(max(score, 0.0), 100.0)
        
    return 0.0


# ==========================================
# 3. Single Pass Ollama Call
# ==========================================
def fetch_single_pass_score(query: str, ground_truth: str, prediction: str) -> float:
    """Executes a single evaluation pass against local Ollama."""
    prompt = f"""
You are an expert evaluator assessing the performance of a Vision-Language Model (VLM).
Evaluate the VLM Prediction against the Ground Truth answer for the given Question/Query.

User Question: "{str(query)[:400]}"
Ground Truth Answer: "{str(ground_truth)[:1500]}"
VLM Prediction: "{str(prediction)[:1500]}"

Grading Scale:
- 90-100: Perfectly answers the question with correct factual meaning (synonyms/paraphrasing accepted).
- 80-89: Highly accurate answer; negligible phrasing or formatting differences.
- 70-79: Mostly correct answer, but misses a minor secondary detail.
- 60-69: Partially correct; addresses the prompt but contains minor ambiguity.
- 50-59: Moderately relevant, but contains minor factual inaccuracies or partial errors.
- 40-49: Addresses the question topic, but misses key target information.
- 30-39: Poor accuracy; contains major factual errors or misinterprets the question.
- 20-29: Highly inaccurate; minimal semantic relevance to the question.
- 10-19: Severe hallucination or completely answers a different question.
- 0-9: Completely wrong, non-sensical, or irrelevant output.

IMPORTANT: Return ONLY the numerical score as a single integer (e.g., 85) at the very end.
"""
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={
                "temperature": 0.2,   # Slight variation across N=3 passes
                "num_predict": 512,  # Cap generation tokens for maximum speed
            },
        )
        raw_output = response["response"].strip()
        return extract_score_from_deepseek(raw_output)

    except Exception as e:
        print(f"\n[Warning] Ollama call failed: {e}")
        return 0.0


# ==========================================
# 4. Multithread Task Processing (N=3 Passes in Parallel)
# ==========================================
def process_row_parallel(row_tuple):
    """Processes all 3 passes for a single row concurrently using sub-threads."""
    idx, row = row_tuple
    q = row["Evaluation Prompt"]
    gt = row["Baseline Output"]       # Baseline output as reference ground truth
    pred = row["Augmented Output"]    # Augmented output as prediction

    if pd.isna(pred) or str(pred).strip() == "" or pd.isna(gt):
        return idx, 0.0

    # Execute all 3 passes in parallel for this row
    with concurrent.futures.ThreadPoolExecutor(max_workers=RUNS_PER_EVAL) as inner_executor:
        pass_futures = [
            inner_executor.submit(fetch_single_pass_score, q, gt, pred)
            for _ in range(RUNS_PER_EVAL)
        ]
        scores = [f.result() for f in pass_futures]

    final_score = float(np.mean(scores)) if scores else 0.0
    return idx, round(final_score, 2)


# ==========================================
# 5. Dataset Processing Function
# ==========================================
def generate_correctness_dataset():
    print(f"Loading master input dataset: {INPUT_FILE}")
    out_df = pd.read_csv(INPUT_FILE, engine="python", encoding="utf-8")

    if "Correctness Score" not in out_df.columns:
        out_df["Correctness Score"] = np.nan

    # Safe Resume Logic: Load progress if output file exists
    if os.path.exists(OUTPUT_FILE):
        print(f"Loading existing checkpoint file '{OUTPUT_FILE}'...")
        existing_df = pd.read_csv(OUTPUT_FILE, engine="python", encoding="utf-8")
        if "Correctness Score" in existing_df.columns:
            for idx in existing_df.index:
                if idx < len(out_df) and not pd.isna(existing_df.at[idx, "Correctness Score"]):
                    out_df.at[idx, "Correctness Score"] = existing_df.at[idx, "Correctness Score"]

    missing_indices = out_df[out_df["Correctness Score"].isna()].index.tolist()
    print(f"Total master rows: {len(out_df)} | Rows remaining to process: {len(missing_indices)}")

    if not missing_indices:
        print("All 1,750 rows are already evaluated!")
        return

    tasks = [(idx, out_df.loc[idx]) for idx in missing_indices]

    print(f"Launching parallel evaluations ({NUM_WORKERS} workers, N={RUNS_PER_EVAL} passes/row)...")

    completed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_row_parallel, task): task[0] for task in tasks}

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(tasks),
            desc="Evaluating Correctness",
        ):
            idx, score = future.result()
            out_df.at[idx, "Correctness Score"] = score
            completed_count += 1

            # Save progress every SAVE_INTERVAL rows
            if completed_count % SAVE_INTERVAL == 0 or completed_count == len(tasks):
                out_df.to_csv(OUTPUT_FILE, index=False)

    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n Dataset successfully saved to '{OUTPUT_FILE}'.")


# ==========================================
# 6. Script Execution
# ==========================================
if __name__ == "__main__":
    generate_correctness_dataset()