"""
Core hoodie classifier implementing 2-stage logic.
"""

import numpy as np
import yaml
from typing import Dict, Any, Tuple
from PIL import Image

from model_loader import create_model_loader
from prompts import get_all_prompts
from utils import (
    normalize_embeddings, 
    sharpen_embeddings, 
    apply_temperature_scaling,
    compute_margin,
    cosine_similarity
)

class HoodieClassifier:
    """Main classifier for hoodie piece classification."""
    
    def __init__(self, config_input = "config.yaml"):
        """Initialize classifier with configuration.
        
        Args:
            config_input: Either a config file path (str) or a config dictionary
        """
        # Set random seeds for deterministic runs
        import random
        
        seed = 42  # Fixed seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        if isinstance(config_input, str):
            # Load config from file
            with open(config_input, 'r') as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config_input, dict):
            # Use provided config dictionary
            self.config = config_input
        else:
            raise ValueError("config_input must be a string (file path) or dictionary")
        
        self.model_loader = create_model_loader(self.config)
        self.prompts = get_all_prompts()
        
        # Load reference embeddings if they exist
        self.ref_embeddings = None
        self.ref_labels = None
        self._load_reference_embeddings()
    
    def _load_reference_embeddings(self):
        """Load pre-computed reference embeddings."""
        import os
        try:
            ref_emb_path = self.config.get("ref_embeddings_path", "ref_embeddings.npy")
            ref_labels_path = self.config.get("ref_labels_path", "ref_labels.npy")
            
            if os.path.exists(ref_emb_path) and os.path.exists(ref_labels_path):
                self.ref_embeddings = np.load(ref_emb_path)
                self.ref_labels = np.load(ref_labels_path)
                print(f"Loaded {len(self.ref_labels)} reference embeddings")
        except Exception as e:
            print(f"Warning: Could not load reference embeddings: {e}")
    
    def classify(self, image_path: str) -> Dict[str, Any]:
        """Classify a hoodie image."""
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.config.get("image_size", 224), self.config.get("image_size", 224)))
        
        # Stage 1: CLIP zero-shot classification
        scores_2, scores_3 = self._stage1_clip_classification(img)
        
        # Compute final scores and margin
        final_score_2 = np.mean(scores_2)
        final_score_3 = np.mean(scores_3)
        margin = compute_margin(final_score_2, final_score_3)
        
        # Convert raw scores to probabilities (0-1 range)
        # Use softmax to ensure scores sum to 1 and are properly normalized
        raw_scores = np.array([final_score_2, final_score_3])
        exp_scores = np.exp(raw_scores)
        probabilities = exp_scores / np.sum(exp_scores)
        
        prob_2 = float(probabilities[0])
        prob_3 = float(probabilities[1])
        
        # Determine if fallback is needed
        margin_threshold = self.config.get("margin_threshold", 0.15)
        fallback_used = False
        
        if margin < margin_threshold and self.ref_embeddings is not None:
            # Stage 2: Reference embedding fallback
            predicted_class = self._stage2_reference_fallback(img)
            fallback_used = True
            
            # Get fallback confidence scores
            fallback_scores = self._get_fallback_confidence(img, predicted_class)
            final_prob_2 = fallback_scores["2"]
            final_prob_3 = fallback_scores["3"]
        else:
            # Use CLIP results
            predicted_class = "2" if final_score_2 > final_score_3 else "3"
            final_prob_2 = prob_2
            final_prob_3 = prob_3
        
        return {
            "pieces": int(predicted_class),
            "scores": {
                "2": final_prob_2,
                "3": final_prob_3
            },
            "margin": float(margin),
            "fallback_used": fallback_used
        }
    
    def _stage1_clip_classification(self, img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """Stage 1: CLIP zero-shot classification with prompt ensemble."""
        # Get image embedding
        img_embedding = self.model_loader.embed_image(img)
        
        # Get text embeddings for all prompts
        all_prompts = []
        prompt_labels = []
        
        for label, prompts in self.prompts.items():
            all_prompts.extend(prompts)
            prompt_labels.extend([label] * len(prompts))
        
        text_embeddings = self.model_loader.embed_text(all_prompts)
        
        # Normalize and sharpen embeddings
        if self.config.get("normalize", True):
            img_embedding = normalize_embeddings(img_embedding)
            text_embeddings = normalize_embeddings(text_embeddings)
        
        exponent = self.config.get("exponent", 1.5)
        img_embedding = sharpen_embeddings(img_embedding, exponent)
        text_embeddings = sharpen_embeddings(text_embeddings, exponent)
        
        # Compute similarities
        similarities = []
        for text_emb in text_embeddings:
            sim = cosine_similarity(img_embedding.flatten(), text_emb)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # Apply temperature scaling
        temperature = self.config.get("temperature", 0.07)
        similarities = apply_temperature_scaling(similarities, temperature)
        
        # Separate scores by class
        scores_2 = []
        scores_3 = []
        
        for i, label in enumerate(prompt_labels):
            if label == "2":
                scores_2.append(similarities[i])
            else:
                scores_3.append(similarities[i])
        
        return np.array(scores_2), np.array(scores_3)
    
    def _stage2_reference_fallback(self, img: Image.Image) -> str:
        """Stage 2: Reference embedding nearest neighbor fallback."""
        if self.ref_embeddings is None:
            raise RuntimeError("Reference embeddings not available")
        
        # Get image embedding
        img_embedding = self.model_loader.embed_image(img)
        
        # Normalize and sharpen - FIX: Don't modify self.ref_embeddings in place!
        if self.config.get("normalize", True):
            img_embedding = normalize_embeddings(img_embedding)
            ref_embeddings = normalize_embeddings(self.ref_embeddings.copy())  # Use copy!
        else:
            ref_embeddings = self.ref_embeddings.copy()  # Use copy!
        
        exponent = self.config.get("exponent", 1.5)
        img_embedding = sharpen_embeddings(img_embedding, exponent)
        ref_embeddings = sharpen_embeddings(ref_embeddings, exponent)
        
        # Find nearest neighbor
        similarities = []
        for ref_emb in ref_embeddings:
            sim = cosine_similarity(img_embedding.flatten(), ref_emb)
            similarities.append(sim)
        
        nearest_idx = np.argmax(similarities)
        return str(self.ref_labels[nearest_idx])
    
    def _get_fallback_confidence(self, img: Image.Image, predicted_class: str) -> Dict[str, float]:
        """Get confidence scores from the fallback method."""
        try:
            # Get image embedding
            img_embedding = self.model_loader.embed_image(img)
            
            # Normalize and sharpen
            if self.config.get("normalize", True):
                img_embedding = normalize_embeddings(img_embedding)
                ref_embeddings = normalize_embeddings(self.ref_embeddings.copy())
            else:
                ref_embeddings = self.ref_embeddings.copy()
            
            exponent = self.config.get("exponent", 1.5)
            img_embedding = sharpen_embeddings(img_embedding, exponent)
            ref_embeddings = sharpen_embeddings(ref_embeddings, exponent)
            
            # Calculate similarities to all reference embeddings
            similarities = []
            for ref_emb in ref_embeddings:
                sim = cosine_similarity(img_embedding.flatten(), ref_emb)
                similarities.append(sim)
            
            # Separate similarities by class
            class_2_similarities = []
            class_3_similarities = []
            
            for i, label in enumerate(self.ref_labels):
                if label == 2:
                    class_2_similarities.append(similarities[i])
                else:
                    class_3_similarities.append(similarities[i])
            
            # Calculate confidence scores based on similarity distributions
            if class_2_similarities and class_3_similarities:
                # Use the maximum similarity for each class as confidence
                max_sim_2 = max(class_2_similarities) if class_2_similarities else 0
                max_sim_3 = max(class_3_similarities) if class_3_similarities else 0
                
                # Normalize to probabilities
                total_sim = max_sim_2 + max_sim_3
                if total_sim > 0:
                    prob_2 = max_sim_2 / total_sim
                    prob_3 = max_sim_3 / total_sim
                else:
                    prob_2 = prob_3 = 0.5
            else:
                # Fallback to equal probabilities if no references
                prob_2 = prob_3 = 0.5
            
            return {"2": float(prob_2), "3": float(prob_3)}
            
        except Exception as e:
            print(f"Warning: Could not get fallback confidence: {e}")
            # Return equal probabilities as fallback
            return {"2": 0.5, "3": 0.5}
    
    def generate_reference_embeddings(self, images_dir: str):
        """Generate reference embeddings from the images directory."""
        import os
        from glob import glob
        
        # Find all hoodie images
        image_files = []
        image_labels = []
        
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob(os.path.join(images_dir, ext)))
        
        # Sort and assign labels based on filename
        image_files.sort()
        for img_path in image_files:
            if '2piece' in img_path.lower():
                image_labels.append(2)
            elif '3piece' in img_path.lower():
                image_labels.append(3)
            else:
                print(f"Warning: Could not determine label for {img_path}")
                continue
        
        # Generate embeddings
        embeddings = []
        labels = []
        
        for img_path, label in zip(image_files, image_labels):
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((self.config.get("image_size", 224), self.config.get("image_size", 224)))
                
                embedding = self.model_loader.embed_image(img)
                embeddings.append(embedding.flatten())
                labels.append(label)
                
                print(f"Generated embedding for {img_path} (label: {label})")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Save embeddings
        if embeddings:
            embeddings = np.array(embeddings)
            labels = np.array(labels)
            
            np.save(self.config.get("ref_embeddings_path", "ref_embeddings.npy"), embeddings)
            np.save(self.config.get("ref_labels_path", "ref_labels.npy"), labels)
            
            print(f"Saved {len(embeddings)} reference embeddings")
            self.ref_embeddings = embeddings
            self.ref_labels = labels
