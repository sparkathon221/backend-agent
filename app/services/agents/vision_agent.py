from crewai import Agent
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import io
import base64
import requests
import json
from sentence_transformers import SentenceTransformer
import os

class VisionAgent:
    def __init__(self):
        self.agent = Agent(
            role="Computer Vision Analyst",
            goal="Extract information from product images and generate embeddings",
            backstory=(
                "Specializes in understanding product images using Qwen Vision model "
                "the recommendations may be related to images or its scenerio"
                "for better recommendations and semantic understanding."
            ),
            allow_delegation=False,
            verbose=True
        )
        # Initialize embedding model for converting descriptions to vectors
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.api_key = os.getenv("HF_TOKEN")
        if not self.api_key:
            raise ValueError("HF_TOKEN environment variable is required")
    
    def process_image(self, image_bytes: bytes) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Process image using Qwen2.5-VL-7B-Instruct model"""
        try:
            # Convert image bytes to base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Generate detailed description using Qwen Vision
            description = self._analyze_image_with_qwen(image_base64)
            
            if description:
                # Convert description to embedding vector
                embedding = self.embedding_model.encode([description], normalize_embeddings=True)
                embedding_vector = np.array(embedding[0]).astype('float32')
                return description, embedding_vector
            else:
                return None, None
                
        except Exception as e:
            print(f"Image processing error: {str(e)}")
            return None, None
    
    def _analyze_image_with_qwen(self, image_base64: str) -> Optional[str]:
        """Analyze image using Qwen2.5-VL-7B-Instruct model"""
        try:
            # Hugging Face Inference API endpoint
            url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-VL-7B-Instruct"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare the payload for vision-language model
            payload = {
                "inputs": {
                    "text": "Describe this product as needed according to the image in detail. Focus on: product type, color, style, material, brand if visible, key features, and any text/labels visible. This will be used for product recommendations.",
                    "image": image_base64
                },
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                # Extract generated text from response
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
                elif isinstance(result, dict):
                    return result.get("generated_text", "").strip()
                else:
                    print(f"Unexpected response format: {result}")
                    return None
            else:
                print(f"Qwen API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Qwen vision analysis error: {str(e)}")
            return None
    
    def extract_features_from_description(self, description: str) -> dict:
        """Extract structured features from image description"""
        try:
            # Use Qwen to extract structured information
            feature_prompt = f"""
            From this product description, extract the following information in JSON format:
            {{
                "product_type": "category of product",
                "color": "main color",
                "style": "style/design",
                "material": "material if mentioned",
                "brand": "brand if visible",
                "key_features": ["feature1", "feature2", "feature3"]
            }}
            
            Description: {description}
            """
            
            url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-VL-7B-Instruct"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": feature_prompt,
                "parameters": {
                    "max_new_tokens": 150,
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                generated_text = ""
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "")
                elif isinstance(result, dict):
                    generated_text = result.get("generated_text", "")
                
                # Try to parse JSON from the response
                try:
                    # Find JSON-like content in the response
                    start_idx = generated_text.find('{')
                    end_idx = generated_text.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        json_str = generated_text[start_idx:end_idx]
                        return json.loads(json_str)
                except:
                    pass
            
            return {}
            
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return {}