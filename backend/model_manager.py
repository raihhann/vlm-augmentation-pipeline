"""Manage preloading of locally hosted Ollama models for inference requests."""

import ollama

class ModelManager:
    """Manages the lifecycle, preloading, and caching of local Ollama vision models."""

    def __init__(self):
        # Dictionary to store loaded models
        self.models = {}

    def load_model(self, model_name: str):
        """Preload a specified Ollama model into memory with an indefinite keep-alive tag.

        Args:
            model_name (str): The name of the model to preload.
        """
        print(f"🚀 Preloading {model_name} into memory...")
        # Sending an empty chat request with keep_alive=-1 loads the model
        ollama.chat(model=model_name, messages=[], keep_alive=-1)
        
        # Track that the model has been successfully loaded
        self.models[model_name] = True
        print(f"✅ {model_name} is ready!")

    def get_model(self, model_name: str):
        """Retrieve a loaded model, preloading it first if not already cached.

        Args:
            model_name (str): The name of the model to retrieve.

        Returns:
            bool | Any: The status or instance object representing the loaded model.
        """
        if model_name not in self.models:
            self.load_model(model_name)

        return self.models[model_name]