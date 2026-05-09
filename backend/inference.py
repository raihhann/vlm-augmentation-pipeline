# inference.py
import ollama
from io import BytesIO
from PIL import Image

def run_inference(model: str, image, prompt: str , type: str) -> str:
    print(f"Running inference with model: {model}, prompt: {prompt}, type: {type}")
    model_lower = model.lower()
    
    # 1. Define the System Instruction for Augmented mode
    # This helps the model understand the relationship between the two images.
    system_instruction = (
        "You are a vision analysis system. The input is a vertical composite of two frames. "
        "FRAME 1 (TOP): This is the high-fidelity original image. Use ONLY this frame for "
        "determining colors, textures, and marking center of focus. It is the most reliable source for your final answer. "
        "FRAME 2 (BOTTOM): This is an augmented version designed for spatial focus. "
        "Use this ONLY to identify the region of interest. "
        "CRITICAL: If visual data in Frame 2 contradicts Frame 1 (due to artificial colors "
        "or artifacts), you MUST defer entirely to Frame 1 for your final answer."
    )

    # --- Pre-process image if it is a PIL object ---
    if isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_input = buffered.getvalue()
    else:
        image_input = image

    if "llava-phi3" in model_lower:
        try:
            # 2. Build the message list dynamically
            messages = []
            
            if type == "augmented":
                # Add the system context for the augmented logic
                messages.append({'role': 'system', 'content': system_instruction})
                # Add the user question
                messages.append({
                    'role': 'user', 
                    'content': f"Based on the provided composite image, {prompt}", 
                    'images': [image_input]
                })
            else:
                # Standard inference
                messages.append({
                    'role': 'user', 
                    'content': prompt, 
                    'images': [image_input]
                })

            response = ollama.chat(
                model=model,
                messages=messages,
            )
            
            print("Finished LLaVA inference of type:", type)
            return response['message']['content']
            
        except Exception as e:
            return f"LLaVA Inference Error: {e}"

    return "Model not supported."