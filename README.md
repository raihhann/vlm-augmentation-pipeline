# Image Augmentation Evaluation Pipeline for Vision-Language Models

This repository contains a research-oriented evaluation pipeline for multimodal vision-language systems. It is part of a **ongoing** research project at the University of Stuttgart and includes both standard and custom proposed augmentation methods. This repo is not the final version of the software it is still work in progress

## Overview

The software allows users to:
- upload images or videos
- select a vision-language model and an augmentation method
- run inference on the original and augmented inputs
- compare outputs, timing, and evaluation metrics
- download results for offline analysis

## Features

- FastAPI backend with streaming progress updates
- web-based frontend for easy interaction
- support for image and video uploads
- evaluation of original vs augmented model outputs
- download report generation
- custom proposed augmentations developed as part of the research

## Proposed Augmentation Methods

The project includes several augmentation modules, including user-proposed methods designed to test how visual transformations affect model performance. These augmentations are implemented under `backend/Augmentify/augment/`.

## Installation

1. Create a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the backend:
   ```bash
   uvicorn backend.main:app --reload
   ```
4. Open the web UI in your browser at `http://127.0.0.1:8000`.

## Project Context

This tool is intended for academic research at Uni Stuttgart, focusing on image augmentation techniques and their effects on vision-language model evaluation. It is designed to support experimentation and comparison of augmentation-driven model behavior.

## Repository Structure

- `backend/` — FastAPI app, augmentation modules, inference and evaluation logic
- `frontend/` — HTML templates for the user interface
- `static/` — shared assets and upload storage
- `requirements.txt` — Python dependencies

## Notes

- The project is a research prototype and may require model configuration for full functionality.
- Augmentation methods in `backend/Augmentify/augment/` include both standard techniques and your own proposed research methods.

## License

This repository is for research use and documentation of a University of Stuttgart project.
