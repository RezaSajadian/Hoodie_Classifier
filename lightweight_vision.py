#!/usr/bin/env python3
"""
Lightweight Vision Analysis using Pure Computer Vision
Provides local vision analysis without CLIP or text - pure image-to-image comparison.
"""

import numpy as np
from PIL import Image
from typing import Dict, Any
import cv2
import glob


class LightweightVisionAnalyzer:
    """Lightweight vision analysis using pure CV technics."""
    
    def __init__(self, config: dict):
        """Initialize lightweight analyzer."""
        self.config = config
    
    def detect_hood_region(self, image: Image.Image) -> Dict[str, Any]:
        """Detect hood region using computer vision techniques."""
        try:
            # Convert to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour (likely the hoodie)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Hood region is typically in the upper portion
                hood_height = int(h * 0.4)  # Top 40%
                
                return {
                    "hood_region": {
                        "x1": x / image.width,
                        "y1": y / image.height,
                        "x2": (x + w) / image.width,
                        "y2": (y + hood_height) / image.height
                    },
                    "confidence": 0.7,
                    "reasoning": "Computer vision edge detection + contour analysis",
                    "local": True
                }
            
            return self._get_default_hood_region(image)
            
        except Exception as e:
            print(f"Error in hood detection: {e}")
            return self._get_default_hood_region(image)
    
    def classify_hood(self, hood_image: Image.Image, full_image: Image.Image = None) -> Dict[str, Any]:
        """Classify hoodie construction using pure CV techniques (no CLIP, no text)."""
        try:
            # Convert to OpenCV format for CV analysis
            hood_cv = cv2.cvtColor(np.array(hood_image), cv2.COLOR_RGB2BGR)
            
            # Extract multiple CV features
            features = self._extract_cv_features(hood_cv)
            
            # Compare with reference images from each class
            class_scores = self._compare_with_references(features)
            
            # Determine classification based on CV similarity
            if class_scores["2-piece"] > class_scores["3-piece"]:
                classification = "2-piece"
                confidence = float(class_scores["2-piece"])
            else:
                classification = "3-piece"
                confidence = float(class_scores["3-piece"])
            
            return {
                "classification": classification,
                "confidence": confidence,
                "reasoning": f"Pure CV analysis: 2-piece={class_scores['2-piece']:.3f}, 3-piece={class_scores['3-piece']:.3f}",
                "key_features": ["Edge patterns", "Texture analysis", "Structural similarity", "Color histograms"],
                "seam_analysis": f"CV-based classification using {classification} construction patterns",
                "local": True
            }
            
        except Exception as e:
            print(f"Error in CV-based classification: {e}")
            return self._get_fallback_classification()
    
    def _extract_cv_features(self, image_cv):
        """Extract comprehensive CV features from hoodie image."""
        features = {}
        
        # 1. Edge detection and analysis
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        features['edges'] = edges
        
        # 2. Texture analysis using Local Binary Patterns
        features['texture'] = self._extract_lbp_features(gray)
        
        # 3. Color histogram analysis
        features['color_hist'] = self._extract_color_histogram(image_cv)
        
        # 4. Structural similarity features
        features['structural'] = self._extract_structural_features(gray)
        
        # 5. Contour and shape analysis
        features['contours'] = self._extract_contour_features(edges)
        
        return features
    
    def _extract_lbp_features(self, gray_image):
        """Extract Local Binary Pattern features for texture analysis."""
        try:
            from skimage.feature import local_binary_pattern
            
            # Calculate LBP
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
            
            # Calculate histogram
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            
            return hist
        except:
            # Fallback if skimage not available
            return np.array([])
    
    def _extract_color_histogram(self, image_cv):
        """Extract color histogram features."""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        return np.concatenate([h_hist, s_hist, v_hist])
    
    def _extract_structural_features(self, gray_image):
        """Extract structural features using Gabor filters."""
        features = []
        
        # Apply Gabor filters at different orientations
        for angle in [0, 45, 90, 135]:
            kernel = cv2.getGaborKernel((21, 21), 8.0, np.radians(angle), 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray_image, cv2.CV_8UC3, kernel)
            features.append(np.mean(filtered))
            features.append(np.std(filtered))
        
        return np.array(features)
    
    def _extract_contour_features(self, edges):
        """Extract contour-based features."""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.array([])
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Extract features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        
        # Shape features
        solidity = float(area) / hull_area if hull_area > 0 else 0
        aspect_ratio = float(cv2.boundingRect(largest_contour)[2]) / cv2.boundingRect(largest_contour)[3] if cv2.boundingRect(largest_contour)[3] > 0 else 0
        
        return np.array([area, perimeter, solidity, aspect_ratio])
    
    def _compare_with_references(self, features):
        """Compare extracted features with reference images from each class."""
        try:
            # Load reference images from images folder
            ref_2piece = self._load_reference_images("2piece")
            ref_3piece = self._load_reference_images("3piece")
            
            # Calculate similarities for each class
            scores_2piece = self._calculate_similarity_scores(features, ref_2piece)
            scores_3piece = self._calculate_similarity_scores(features, ref_3piece)
            
            # Return average scores
            return {
                "2-piece": np.mean(scores_2piece) if scores_2piece else 0.5,
                "3-piece": np.mean(scores_3piece) if scores_3piece else 0.5
            }
            
        except Exception as e:
            print(f"Error in reference comparison: {e}")
            return {"2-piece": 0.5, "3-piece": 0.5}
    
    def _load_reference_images(self, class_type):
        """Load reference images for a specific class."""
        ref_images = []
        pattern = f"images/hoodie_{class_type}_*.png"
        
        for img_path in glob.glob(pattern):
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    ref_images.append(img)
            except:
                continue
        
        return ref_images
    
    def _calculate_similarity_scores(self, features, reference_images):
        """Calculate similarity scores between features and reference images."""
        if not reference_images:
            return []
        
        scores = []
        
        for ref_img in reference_images:
            # Extract features from reference image
            ref_features = self._extract_cv_features(ref_img)
            
            # Calculate similarity score
            score = self._compute_similarity(features, ref_features)
            scores.append(score)
        
        return scores
    
    def _compute_similarity(self, features1, features2):
        """Compute overall similarity between two feature sets."""
        total_score = 0
        weights = {
            'edges': 0.3,
            'texture': 0.25,
            'color_hist': 0.2,
            'structural': 0.15,
            'contours': 0.1
        }
        
        # Edge similarity
        if 'edges' in features1 and 'edges' in features2:
            edge_sim = self._compare_edges(features1['edges'], features2['edges'])
            total_score += weights['edges'] * edge_sim
        
        # Texture similarity
        if 'texture' in features1 and 'texture' in features2:
            texture_sim = self._compare_histograms(features1['texture'], features2['texture'])
            total_score += weights['texture'] * texture_sim
        
        # Color histogram similarity
        if 'color_hist' in features1 and 'color_hist' in features2:
            color_sim = self._compare_histograms(features1['color_hist'], features2['color_hist'])
            total_score += weights['color_hist'] * color_sim
        
        # Structural similarity
        if 'structural' in features1 and 'structural' in features2:
            struct_sim = self._compare_arrays(features1['structural'], features2['structural'])
            total_score += weights['structural'] * struct_sim
        
        # Contour similarity
        if 'contours' in features1 and 'contours' in features2:
            contour_sim = self._compare_arrays(features1['contours'], features2['contours'])
            total_score += weights['contours'] * contour_sim
        
        return total_score
    
    def _compare_edges(self, edges1, edges2):
        """Compare edge patterns using structural similarity."""
        try:
            from skimage.metrics import structural_similarity as ssim
            # Ensure same dimensions for comparison
            if edges1.shape != edges2.shape:
                # Resize edges2 to match edges1
                edges2_resized = cv2.resize(edges2, (edges1.shape[1], edges1.shape[0]))
                return ssim(edges1, edges2_resized)
            return ssim(edges1, edges2)
        except:
            # Fallback: simple correlation with resizing
            if edges1.shape != edges2.shape:
                edges2_resized = cv2.resize(edges2, (edges1.shape[1], edges1.shape[0]))
                edges2 = edges2_resized
            return np.corrcoef(edges1.flatten(), edges2.flatten())[0, 1] if edges1.size > 0 and edges2.size > 0 else 0
    
    def _compare_histograms(self, hist1, hist2):
        """Compare histograms using correlation."""
        if hist1.size == 0 or hist2.size == 0:
            return 0
        
        # Ensure same length
        min_len = min(len(hist1), len(hist2))
        hist1 = hist1[:min_len]
        hist2 = hist2[:min_len]
        
        return np.corrcoef(hist1, hist2)[0, 1]
    
    def _compare_arrays(self, arr1, arr2):
        """Compare arrays using cosine similarity."""
        if arr1.size == 0 or arr2.size == 0:
            return 0
        
        # Ensure same length
        min_len = min(len(arr1), len(arr2))
        arr1 = arr1[:min_len]
        arr2 = arr2[:min_len]
        
        # Cosine similarity
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        return 0
    
    def _get_default_hood_region(self, image: Image.Image) -> Dict[str, Any]:
        """Default hood region (top 40% of image)."""
        return {
            "hood_region": {
                "x1": 0.0,
                "y1": 0.0,
                "x2": 1.0,
                "y2": 0.4
            },
            "confidence": 0.5,
            "reasoning": "Fallback to default top region",
            "local": True
        }
    
    def _get_fallback_classification(self) -> Dict[str, Any]:
        """Fallback classification when analysis fails."""
        return {
            "classification": "unknown",
            "confidence": 0.0,
            "reasoning": "Lightweight vision analysis failed",
            "key_features": [],
            "seam_analysis": "No analysis available",
            "local": True
        }



