"""Manage preloading of locally hosted Ollama models for inference requests."""

import ollama

class ModelManager:
    def __init__(self):
        # Dictionary to store loaded models
        self.models = {}

    def load_model(self, model_name: str):
        print(f"🚀 Preloading {model_name} into memory...")
        # Sending an empty chat request with keep_alive=-1 loads the model
        ollama.chat(model=model_name, messages=[], keep_alive=-1)
        print(f"✅ {model_name} is ready!")

    def get_model(self, model_name: str):
        """
        Returns loaded model.
        If not loaded yet, load it first.
        """
        if model_name not in self.models:
            self.load_model(model_name)

        return self.models[model_name]