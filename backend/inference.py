"""Run vision-language model inference through the local Ollama service.

The public helper accepts an image path and prompt, handles the supported model
selection, and returns the generated textual description.
"""

import os
from PIL import Image
import ollama


def run_inference(
    model: str, image_input: str, prompt: str, type: str
) -> str:
  print(f"Running inference with model: {model}, type: {type}")

  # ==========================================
  # 🧪 MOCK TEST FALLBACK SUBORDINATOR
  # ==========================================
  if str(model).lower().startswith("mock"):
    import random
    import time

    time.sleep(0.5)

    mock_scenarios = [
        f"[MOCK VLM RESPONSE ({type.upper()} MODE)]\nAnalysis of the tracking"
        f" sequence relative to prompt: '{prompt}'.\nThe model successfully"
        " isolated the focal object moving across the frame bounding zones."
        " The subject maintains a stable vector trajectory despite localized"
        " lighting noise.",
        f"[MOCK VLM RESPONSE ({type.upper()} MODE)]\nThe vision system reports"
        " high structural matching confidence for the target object. Temporal"
        " redundancy filtering has eliminated background drift artifacts,"
        " leaving clear semantic observations.",
    ]
    return random.choice(mock_scenarios)

  # ==========================================
  # 📷 STANDARD LIVE INFERENCE PIPELINE
  # ==========================================
  system_instruction = (
      "You are a specialized vision analysis system. Inputs are a vertical"
      " composite of two frames. FRAME 1 (TOP): The high-fidelity original"
      " image. Use ONLY this frame for identifying colors, textures, and breed"
      " characteristics. FRAME 2 (BOTTOM): An augmented version for spatial"
      " focus. Use this ONLY to identify the region of interest. Defer"
      " entirely to Frame 1 for final answers."
  )

  # 1. Resolve Path to Local System File
  if isinstance(image_input, Image.Image):
    path_to_send = "temp_inference_image.jpg"
    image_input.save(path_to_send, format="JPEG", quality=95)
  elif isinstance(image_input, str):
    # Convert web paths like "/static/uploads/1.png" -> "../static/uploads/1.png"
    if image_input.startswith("/static/"):
      path_to_send = image_input.replace("/static/", "../static/", 1)
    else:
      path_to_send = image_input
  else:
    path_to_send = str(image_input)

  # Absolute path resolution to prevent any directory mismatches
  path_to_send = os.path.abspath(path_to_send)

  # Sanity check: Ensure file actually exists before sending to Ollama
  if not os.path.exists(path_to_send):
    return f"Inference Error: Image file not found on disk at '{path_to_send}'"

  # 2. Build messages
  messages = []
  if type == "augmented":
    messages.append({"role": "system", "content": system_instruction})
    user_content = f"Based on the provided composite image, {prompt}"
  else:
    user_content = prompt

  messages.append({
      "role": "user",
      "content": user_content,
      "images": [path_to_send],  # Pass verified absolute path
  })

  try:
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]
  except Exception as e:
    return f"Inference Error: {e}"