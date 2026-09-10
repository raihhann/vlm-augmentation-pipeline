# High-Performance Computing (HPC) Evaluation Scripts

This directory contains backend evaluation scripts designed to be offloaded to a High-Performance Computing (HPC) workstation (equipped with dual NVIDIA RTX 4090 GPUs). These scripts handle resource-intensive semantic correctness scoring and multi-agent debate consensus evaluations asynchronously, preventing compute resource starvation on edge hardware during live mobile robotics inference.

---

## 📂 Script Directory Structure

```text
HPC Scripts/
├── debate_system.py             # Heterogeneous multi-agent debate consensus framework
└── LLM_as_Judge.py             # Automated semantic correctness grading pipeline
```

---

## 🔬 Module Overview

### 1. LLM-as-a-Judge Evaluation (`LLM_as_Judge.py`)
* **Purpose:** Evaluates open-ended textual predictions against ground-truth reference descriptions without relying solely on surface-level word matching.
* **Mechanism:** Leverages an instruction-tuned reasoning model (DeepSeek-R1 32B) to execute step-by-step Chain-of-Thought (CoT) verification across visual-spatial reasoning tasks, parsing internal thinking blocks (`<think>...</think>`) and grading predictions on a ten-tier analytical scale.
* **Execution Protocol:** Executes across $N_{pass} = 3$ parallel evaluation passes per dataset row with low sampling temperature ($T_{sample} = 0.2$) to compute a stable arithmetic mean correctness score ($S_{eval}$).

### 2. Heterogeneous Multi-Agent Debate System (`debate_system.py`)
* **Purpose:** Eliminates position, length, and verbosity bias when evaluating competing visual prompt augmentation methods across large-scale benchmark runs.
* **Multi-Agent Architecture:** Distributes evaluation responsibilities across specialized LLM roles based on model parameter strengths:
  * **Candidate Shortlisting Agent (Qwen2.5:14b):** Filters a pool of 35 candidate outputs down to the Top-5 methods ($K_{shortlist} = 5$) matching ground-truth criteria.
  * **Factuality Critic Agent (DeepSeek-R1 32B):** Inspects shortlisted candidates specifically for visual hallucinations, false statements, or physical contradictions.
  * **Visual Detail Advocate Agent (Llama 3):** Evaluates candidate responses for spatial completeness, structural thoroughness, and descriptive detail.
  * **Chief Judge Agent (Qwen2.5:14b):** Reviews original queries, ground truths, candidate predictions, and written arguments from both the Critic and Advocate to issue a structured JSON verdict.
* **Consensus Protocol:** Repeats the debate workflow across $N_{round} = 3$ independent passes per image, applying majority consensus voting ($M_{winner}$) to determine final method rankings.

---

## 🚀 Execution & Integration

These scripts ingest raw telemetry logs and prediction records generated during edge execution on the NVIDIA Jetson AGX Thor. Ensure your local Ollama environment or local model weights are running on the HPC workstation before executing evaluation passes.