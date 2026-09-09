# Future Work

This directory is reserved for planned research, prototyping, and engineering 
extensions to Augmentify. It now contains active exploratory stubs and 
boilerplate modules for upcoming pipeline upgrades.

## Purpose

Future modules are placed here when their intended behavior needs to be documented 
and prototyped before full production integration. Modules often begin as exploratory 
Python files containing a module-level docstring, functional stubs or fallbacks, 
and clear markers for upcoming enhancements.

## Current Prototype Modules

* **Super Smart Sampling (`super_smart_sampling.py` / sampling stubs):** 
  * *Purpose:* Designed to optimize video keyframe selection for long, multi-scene videos.
  * *Current State:* Implements a baseline/fallback selection mechanism with a roadmap to transition into a query-grounded scoring system that minimizes downstream VLM latency.
* **Adaptive Policy Selection (`run_adaptive_policy_selection`):** 
  * *Purpose:* Dynamically chooses and executes optimal image augmentation policies based on image contents and text prompts.
  * *Current State:* Structured as a functional placeholder ready for advanced policy-routing logic.
* **Lightweight Semantic Evaluation (`evaluate_lightweight_semantics`):** 
  * *Purpose:* Efficiently scores VLM outputs against ground-truth references without heavy embedding bottlenecks.
  * *Current State:* Implements a lightweight heuristic stub returning quick pseudo-scores for fast iterative testing.

---

## Candidate Work Areas

* Complete and stabilize the image augmentation dispatcher so all verified augmenters can be selected safely.
* Add additional proposed augmentations and compare them against established transformations.
* Improve video keyframe selection for long, multi-scene, and query-specific videos using advanced sampling strategies.
* Add batch-run reproducibility through explicit random seeds and persisted experiment configuration.
* Move session storage from process memory to a durable experiment-result store.
* Add validation, error reporting, and cleanup for uploaded media and generated model outputs.
* Expand automated tests for image contracts, extractor edge cases, metrics, and API workflows.
* Improve dashboard comparison views and expose all backend metrics consistently.
* Add experiment metadata, aggregate statistics, and repeat-run confidence analysis.

*Note: These are planning topics and active prototyping areas, not strict promises about current production functionality. Proposals must be validated against current API and model constraints before full implementation.*

---

## Suggested Boilerplate Structure

When introducing a new experimental file, follow this layout:

```python
"""Describe the planned method, research purpose, inputs, outputs, and dependencies.

Implementation status: exploratory / planned.
"""


def run_planned_method(image, **kwargs):
    """Return the documented result or prototype fallback once the method is drafted."""
    print("Executing exploratory placeholder routine...")
    return image
```

When a module becomes fully production-ready, move it to the appropriate core implementation folder, add comprehensive unit tests, register it in the corresponding dispatcher or extractor registry, and update the main documentation files.