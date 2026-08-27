# Future Work

This directory is reserved for planned research and engineering extensions to Augmentify. It is intentionally a documentation and boilerplate area at the moment; no production implementation is currently stored here.

## Purpose

Future modules should be placed here when their intended behavior needs to be documented before implementation. A planned module may begin as a Python file containing a module-level docstring, function signatures, and `NotImplementedError` placeholders. The docstring should explain the research motivation, expected inputs and outputs, dependencies, and open design decisions.

## Candidate work areas

- Complete and stabilize the image augmentation dispatcher so all verified augmenters can be selected safely.
- Add additional proposed augmentations and compare them against established transformations.
- Improve video keyframe selection for long, multi-scene, and query-specific videos.
- Add batch-run reproducibility through explicit random seeds and persisted experiment configuration.
- Move session storage from process memory to a durable experiment-result store.
- Add validation, error reporting, and cleanup for uploaded media and generated model outputs.
- Expand automated tests for image contracts, extractor edge cases, metrics, and API workflows.
- Improve dashboard comparison views and expose all backend metrics consistently.
- Add experiment metadata, aggregate statistics, and repeat-run confidence analysis.

These are planning topics, not promises about current functionality. A proposal should be validated against the current API and model constraints before implementation.

## Suggested boilerplate

A future Python module can start with this structure:

```python
"""Describe the planned method, research purpose, inputs, outputs, and dependencies.

Implementation status: planned.
"""


def run_planned_method(image, **kwargs):
    """Return the documented result once the method is implemented."""
    raise NotImplementedError("This FutureWork method is not implemented yet")
```

When the module becomes usable, move it to the appropriate implementation folder, add focused tests, register it in the dispatcher or extractor registry, and update the relevant README files.
