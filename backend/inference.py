import ollama
import os
from PIL import Image

def run_inference(model: str, image_input, prompt: str, type: str) -> str:
    print(f"Running inference with model: {model}, type: {type}")
    
    # 1. Define the Universal System Instruction
    system_instruction = (
        "You are a specialized vision analysis system. Inputs are a vertical composite of two frames. "
        "FRAME 1 (TOP): The high-fidelity original image. Use ONLY this frame for "
        "identifying colors, textures, and breed characteristics. "
        "FRAME 2 (BOTTOM): An augmented version for spatial focus. Use this ONLY to "
        "identify the region of interest. Defer entirely to Frame 1 for final answers."
    )

    # 2. Handle Image Input (Ensure we have a path or bytes)
    # If the user passed a PIL object, save it to a temp file to ensure clean transfer
    path_to_send = image_input
    if isinstance(image_input, Image.Image):
        path_to_send = "temp_inference_image.jpg"
        image_input.save(path_to_send, format="JPEG", quality=95)
    
    # 3. Build the messages
    messages = []
    if type == "augmented":
        messages.append({'role': 'system', 'content': system_instruction})
        user_content = f"Based on the provided composite image, {prompt}"
    else:
        user_content = prompt

    messages.append({
        'role': 'user',
        'content': user_content,
        'images': [path_to_send] # Ollama SDK accepts the path string directly
    })

    try:
        response = ollama.chat(model=model, messages=messages)
        return response['message']['content']
    except Exception as e:
        return f"Inference Error: {e}"