# inference.py
import ollama
from io import BytesIO
from PIL import Image

def run_inference(model: str, image, prompt: str , type: str) -> str:
    print(f"Running inference with model: {model}, prompt: {prompt}, type: {type}")
    model_lower = model.lower()
    
    if type == "augmented":
        final_prompt = (
            "The provided image is a vertical stack of two versions of the same scene."
            "The top half is the original image, and the bottom half is the augmented version. "
            "Based on this layout, please answer the following: {prompt}"
        )
    # --- Pre-process image if it is a PIL object ---
    if isinstance(image, Image.Image):
        buffered = BytesIO()
        # Convert PIL to JPEG bytes
        image.save(buffered, format="JPEG")
        image_input = buffered.getvalue()
    else:
        # If it's already a string (path), keep it as is
        image_input = image

    if "llava-phi3" in model_lower:
        try:
            response = ollama.chat(
                model=model,
                messages=[{
                    'role': 'user',
                    'content': final_prompt if type == "augmented" else prompt,
                    'images': [image_input] # Ollama accepts path string OR bytes
                }],
            )
            print("Finished LLaVA inference of type:", type)
            return response['message']['content']
        except Exception as e:
            return f"LLaVA Inference Error: {e}"

    return "Model not supported."