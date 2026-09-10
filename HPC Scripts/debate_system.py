import collections
import json
import os
import re
import ollama
import pandas as pd
from tqdm import tqdm

# ==========================================
# 1. Multi-Model Configuration per Agent Role
# ==========================================
SHORTLIST_MODEL = "qwen2.5:14b"     # Fast, reliable JSON parsing
CRITIC_MODEL    = "deepseek-r1:32b" # Deep CoT reasoning for finding flaws
ADVOCATE_MODEL  = "llama3:latest"         # Strong descriptive synthesis (or "llama3")
JUDGE_MODEL     = "qwen2.5:14b"     # Unbiased, strict rubric enforcer

RUNS_PER_IMAGE = 3
TOP_K_CANDIDATES = 5

INPUT_FILE = "INPUT CSV FILE PATH HERE"  # e.g., "debate_dataset.csv"
OUTPUT_EXCEL = "debate_results.xlsx"


def call_ollama(prompt: str, model: str, temperature: float = 0.2) -> str:
    """Helper function to execute Ollama LLM requests for a specific model."""
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature},
        )
        return response["response"].strip()
    except Exception as e:
        print(f"\n[Warning] Ollama call for model '{model}' failed: {e}")
        return ""


# ==========================================
# 2. Stage 1: Candidate Shortlisting
# ==========================================
def shortlist_candidates(
    query: str, ground_truth: str, method_outputs: dict
) -> list:
    """Filters 35 methods down to the Top-K candidates using the SHORTLIST_MODEL."""
    if len(method_outputs) <= TOP_K_CANDIDATES:
        return list(method_outputs.keys())

    formatted_outputs = ""
    for method, pred in method_outputs.items():
        snippet = str(pred)[:300].replace("\n", " ")
        formatted_outputs += f"- Method [{method}]: {snippet}\n"

    prompt = f"""
You are a preliminary judge. Select the top {TOP_K_CANDIDATES} methods whose predictions best match the Ground Truth.

Question: "{query}"
Ground Truth: "{ground_truth}"

Candidates:
{formatted_outputs}

Return ONLY a valid JSON list of the top {TOP_K_CANDIDATES} method names. 
Example format: ["FastSAM", "CLAHE", "MIDAS", "ZoeDepth"]
Do not add any explanation.
"""
    raw_res = call_ollama(prompt, model=SHORTLIST_MODEL, temperature=0.0)
    match = re.search(r"\[.*\]", raw_res, re.DOTALL)
    if match:
        try:
            candidates = json.loads(match.group(0))
            valid_candidates = [c for c in candidates if c in method_outputs]
            if len(valid_candidates) > 0:
                return valid_candidates[:TOP_K_CANDIDATES]
        except Exception:
            pass

    return list(method_outputs.keys())[:TOP_K_CANDIDATES]


# ==========================================
# 3. Stage 2: Heterogeneous Debate Engine
# ==========================================
def run_agentic_debate(
    query: str, ground_truth: str, candidate_outputs: dict
) -> dict:
    """Executes debate using different models for Critic, Advocate, and Judge."""
    candidates_text = ""
    for method, pred in candidate_outputs.items():
        candidates_text += f"=== METHOD: {method} ===\nPrediction: {pred}\n\n"

    # Agent 1: Factuality Critic (e.g., DeepSeek-R1)
    critic_prompt = f"""
You are the Factuality Critic Agent.
Analyze these predictions against the Ground Truth for the query: "{query}"
Ground Truth: "{ground_truth}"

Candidate Predictions:
{candidates_text}

Identify which predictions contain visual hallucinations, contradictions, or false statements relative to the Ground Truth.
Be brief and highlight specific flaws.
"""
    critic_argument = call_ollama(critic_prompt, model=CRITIC_MODEL, temperature=0.3)

    # Agent 2: Visual Detail Advocate (e.g., Llama 3.3)
    advocate_prompt = f"""
You are the Visual Detail Advocate Agent.
Analyze these predictions for completeness, detail, and spatial accuracy regarding: "{query}"
Ground Truth: "{ground_truth}"

Candidate Predictions:
{candidates_text}

Identify which method offers the most thorough, clear, and informative response without losing factual ground.
Be brief and defend the best candidate.
"""
    advocate_argument = call_ollama(advocate_prompt, model=ADVOCATE_MODEL, temperature=0.3)

    # Agent 3: Chief Judge (e.g., Qwen 2.5)
    judge_prompt = f"""
You are the Chief Judge presiding over a VLM evaluation debate.

User Question: "{query}"
Ground Truth Reference: "{ground_truth}"

Candidate Predictions:
{candidates_text}

Factuality Critic Arguments (from {CRITIC_MODEL}):
{critic_argument}

Detail Advocate Arguments (from {ADVOCATE_MODEL}):
{advocate_argument}

Based on the predictions and the debate arguments, determine WHICH SINGLE METHOD is the absolute best.

Output MUST be formatted as a valid JSON object with two keys:
1. "winning_method": exact name of the winning method string.
2. "reasoning": 1-2 sentence summary of why it won the debate.

JSON Output:
"""
    judge_res = call_ollama(judge_prompt, model=JUDGE_MODEL, temperature=0.1)

    # Extract JSON verdict
    match = re.search(r"\{.*\}", judge_res, re.DOTALL)
    if match:
        try:
            verdict = json.loads(match.group(0))
            winner = verdict.get("winning_method", "")
            reasoning = verdict.get("reasoning", "No reasoning provided.")

            for cand in candidate_outputs.keys():
                if cand.lower() in str(winner).lower():
                    return {"winner": cand, "reasoning": reasoning}
        except Exception:
            pass

    fallback_winner = list(candidate_outputs.keys())[0]
    return {
        "winner": fallback_winner,
        "reasoning": "Fallback selection due to parsing format.",
    }


# ==========================================
# 4. Main Processing & Excel Export
# ==========================================
def evaluate_dataset_with_debate(
    input_file: str = INPUT_FILE,
    output_excel: str = OUTPUT_EXCEL,
):
    print(f"Loading dataset: {input_file}")
    df = pd.read_csv(input_file, engine="python", encoding="utf-8")
    grouped = df.groupby(["Image Filename", "Evaluation Prompt"], sort=False)

    results = []
    print(f"Starting Multi-Model Agent Debate across {len(grouped)} unique images...")

    for image_name, group in tqdm(grouped, desc="Processing Images"):
        first_row = group.iloc[0]
        query = first_row.get("Evaluation Prompt", "")
        ground_truth = first_row.get("Ground Truth Reference", "")
        question_cat = first_row.get("Question Category", "")
        is_lowlight = first_row.get("is_lowlight", "")

        method_outputs = {}
        for _, row in group.iterrows():
            method = str(row.get("Evaluation Method", "")).strip()
            pred = str(row.get("Augmented Output", "")).strip()
            if method and pred and pred != "nan":
                method_outputs[method] = pred

        if not method_outputs:
            continue

        # Step 1: Shortlist
        top_candidates = shortlist_candidates(query, ground_truth, method_outputs)
        candidate_outputs = {c: method_outputs[c] for c in top_candidates}

        # Step 2: Debate Loop (N=3 passes)
        run_winners = []
        last_reasoning = ""

        for _ in range(RUNS_PER_IMAGE):
            debate_result = run_agentic_debate(query, ground_truth, candidate_outputs)
            run_winners.append(debate_result["winner"])
            last_reasoning = debate_result["reasoning"]

        # Step 3: Consensus Vote
        vote_counts = collections.Counter(run_winners)
        consensus_winner = vote_counts.most_common(1)[0][0]

        results.append(
            {
                "Image Filename": image_name,
                "Question Category": question_cat,
                "is_lowlight": is_lowlight,
                "Evaluation Prompt": query,
                "Ground Truth Reference": ground_truth,
                "Candidate Pool": ", ".join(top_candidates),
                "Critic Model": CRITIC_MODEL,
                "Advocate Model": ADVOCATE_MODEL,
                "Judge Model": JUDGE_MODEL,
                "Run 1 Winner": run_winners[0],
                "Run 2 Winner": run_winners[1],
                "Run 3 Winner": run_winners[2],
                "Debate Consensus Winner": consensus_winner,
                "Judge Reasoning": last_reasoning,
            }
        )

        if len(results) % 5 == 0:
            pd.DataFrame(results).to_excel(output_excel, index=False)

    final_df = pd.DataFrame(results)
    final_df.to_excel(output_excel, index=False)
    print(f"\nDebate complete! Exported to '{output_excel}'.")


if __name__ == "__main__":
    evaluate_dataset_with_debate()