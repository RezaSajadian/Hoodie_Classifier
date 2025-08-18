"""
Utility functions for image processing and embedding manipulation.
"""

import numpy as np



def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return embeddings / norms

def sharpen_embeddings(embeddings: np.ndarray, exponent: float = 1.5) -> np.ndarray:
    """Apply sharpening to embeddings using the specified exponent."""
    return np.sign(embeddings) * (np.abs(embeddings) ** exponent)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def apply_temperature_scaling(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to scores."""
    return scores / temperature

def compute_margin(score_2: float, score_3: float) -> float:
    """Compute the margin between two scores."""
    return abs(score_2 - score_3)
