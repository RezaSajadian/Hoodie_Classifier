"""
Model loader for different embedding providers.
Supports local OpenCLIP, OpenAI API, and HuggingFace API.
"""

import os
import numpy as np
from typing import List
from PIL import Image


class ModelLoader:
    """Abstract base class for model loaders."""
    
    def embed_text(self, prompt_list: List[str]) -> np.ndarray:
        """Embed a list of text prompts."""
        raise NotImplementedError
    
    def embed_image(self, img: Image.Image) -> np.ndarray:
        """Embed a single image."""
        raise NotImplementedError

class LocalOpenCLIPLoader(ModelLoader):
    """Local OpenCLIP model loader."""
    
    def __init__(self, model_name: str = "ViT-B-32"):
        try:
            import open_clip
            import torch
        except ImportError:
            raise ImportError("OpenCLIP not installed. Run: pip install open_clip_torch")
        
        self.device = "cpu"
        self.model_name = model_name
        
        # Determine the correct pretrained dataset and configuration
        if model_name in ["ViT-B-16", "ViT-B-32", "RN50", "RN101"]:
            pretrained = "openai"
            force_quick_gelu = True  # OpenAI models use QuickGELU
        elif model_name in ["ViT-B-16-quickgelu", "ViT-B-32-quickgelu"]:
            pretrained = "openai"
            force_quick_gelu = True
        else:
            pretrained = "openai"  # Default fallback
            force_quick_gelu = False
        
        print(f"Loading {model_name} with pretrained weights from '{pretrained}' dataset")
        
        # Create model and image transforms
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            force_quick_gelu=force_quick_gelu,
            device=self.device
        )
        
        # Get the text tokenizer separately
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        self.model.eval()
        
        print(f"Successfully loaded {model_name} with pretrained weights")
    
    def embed_text(self, prompt_list: List[str]) -> np.ndarray:
        """Embed text prompts using OpenCLIP."""
        import torch
        
        # Process each prompt separately
        all_features = []
        for prompt in prompt_list:
            tokens = self.tokenizer(prompt).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(tokens)
                all_features.append(text_features.cpu())
        
        # Concatenate all features
        return torch.cat(all_features, dim=0).numpy()
    
    def embed_image(self, img: Image.Image) -> np.ndarray:
        """Embed image using OpenCLIP."""
        import torch
        
        # Use validation preprocess (no augmentation)
        img_tensor = self.preprocess_val(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(img_tensor)
            return image_features.cpu().numpy()

class OpenAIAPILoader(ModelLoader):
    """OpenAI API model loader."""
    
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI not installed. Run: pip install openai")
        
        self.model_name = model_name
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        openai.api_key = api_key
    
    def embed_text(self, prompt_list: List[str]) -> np.ndarray:
        """Embed text prompts using OpenAI API."""
        import openai
        
        embeddings = []
        for prompt in prompt_list:
            response = openai.Embedding.create(
                input=prompt,
                model=self.model_name
            )
            embeddings.append(response['data'][0]['embedding'])
        
        return np.array(embeddings)
    
    def embed_image(self, img: Image.Image) -> np.ndarray:
        """OpenAI API doesn't support image embedding in this model."""
        raise NotImplementedError("OpenAI text-embedding-ada-002 doesn't support image embedding")

class HuggingFaceAPILoader(ModelLoader):
    """HuggingFace API model loader."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            import requests
        except ImportError:
            raise ImportError("Requests not installed. Run: pip install requests")
        
        self.model_name = model_name
        self.api_token = os.getenv("HF_API_TOKEN")
        if not self.api_token:
            raise ValueError("HF_API_TOKEN environment variable not set")
        
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
    
    def embed_text(self, prompt_list: List[str]) -> np.ndarray:
        """Embed text prompts using HuggingFace API."""
        import requests
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": prompt_list}
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"HF API error: {response.status_code} - {response.text}")
        
        return np.array(response.json())
    
    def embed_image(self, img: Image.Image) -> np.ndarray:
        """HuggingFace API doesn't support image embedding in this model."""
        raise NotImplementedError("HuggingFace text model doesn't support image embedding")

def create_model_loader(config: dict) -> ModelLoader:
    """Create model loader based on configuration."""
    provider = config.get("model_provider", "local")
    
    if provider == "local":
        model_name = config.get("model_name", "ViT-B-32")
        return LocalOpenCLIPLoader(model_name)
    elif provider == "openai":
        model_name = config.get("openai_model", "text-embedding-ada-002")
        return OpenAIAPILoader(model_name)
    elif provider == "huggingface-api":
        model_name = config.get("hf_model", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceAPILoader(model_name)
    else:
        raise ValueError(f"Unknown model provider: {provider}")
