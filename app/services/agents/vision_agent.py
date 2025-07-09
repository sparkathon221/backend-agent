from crewai import Agent
from typing import Tuple, Optional, List, Dict
import numpy as np
from PIL import Image
import io
import base64
import requests
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from concurrent.futures import ThreadPoolExecutor

import os
import uuid

from contextlib import contextmanager

@contextmanager
def with_temp_file(content: bytes, ext: str = "jpg", folder: str = "./temp"):
    os.makedirs(folder, exist_ok=True)
    temp_filename = f"{uuid.uuid4().hex}.{ext}"
    temp_path = os.path.join(folder, temp_filename)

    try:
        with open(temp_path, "wb") as f:
            f.write(content)
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

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
        
    def process_image_url(self, image_url: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Process image from URL using Qwen2.5-VL-7B-Instruct model"""
        try:
            # Download image from URL
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image_bytes = response.content
            return self.process_image(image_bytes)
        except Exception as e:
            print(f"Error processing image URL: {str(e)}")
            return None, None
    
    def sort_products_by_relevance(self, 
                                 products: List[Dict],
                                 user_query: str,
                                 image_descriptions: Dict[str, str]) -> List[Dict]:
        """Sort products by relevance based on user query and image descriptions"""
        try:
            # Convert user query to embedding
            query_embedding = self.embedding_model.encode([user_query], normalize_embeddings=True)[0]
            
            # Process products in parallel
            with ThreadPoolExecutor() as executor:
                futures = []
                for product in products:
                    if product['image_url'] in image_descriptions:
                        future = executor.submit(
                            self._calculate_relevance,
                            product,
                            query_embedding,
                            image_descriptions[product['image_url']]
                        )
                        futures.append((product, future))
                
                # Get results and sort by relevance
                products_with_scores = []
                for product, future in futures:
                    try:
                        relevance_score = future.result()
                        products_with_scores.append((product, relevance_score))
                    except Exception as e:
                        print(f"Error processing product {product.get('product_id', 'unknown')}: {str(e)}")
                        products_with_scores.append((product, 0))
                
                # Sort by relevance score (highest first)
                products_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                return [p[0] for p in products_with_scores]
                
        except Exception as e:
            print(f"Error sorting products: {str(e)}")
            return products
    
    def _calculate_relevance(self, 
                            product: Dict,
                            query_embedding: np.ndarray,
                            image_description: str) -> float:
        """Calculate relevance score for a product based on query and image description"""
        try:
            # Generate product description embedding
            product_text = f"{product.get('title', '')} {product.get('brand', '')}"
            product_embedding = self.embedding_model.encode([product_text], normalize_embeddings=True)[0]
            
            # Generate image description embedding
            image_embedding = self.embedding_model.encode([image_description], normalize_embeddings=True)[0]
            
            # Calculate combined relevance using weighted cosine similarity
            query_product_sim = cosine_similarity([query_embedding], [product_embedding])[0][0]
            query_image_sim = cosine_similarity([query_embedding], [image_embedding])[0][0]
            product_image_sim = cosine_similarity([product_embedding], [image_embedding])[0][0]
            
            # Weighted combination of similarities
            # Weights: 0.4 for query-product, 0.4 for query-image, 0.2 for product-image
            relevance_score = (
                0.4 * query_product_sim +
                0.4 * query_image_sim +
                0.2 * product_image_sim
            )
            
            return relevance_score
            
        except Exception as e:
            print(f"Error calculating relevance for product {product.get('product_id', 'unknown')}: {str(e)}")
            return 0
    
    def process_image(self, image_bytes: bytes) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Process image using Qwen2.5-VL-7B-Instruct model"""
        try:
            # Convert image bytes to base64 and create data URI
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            image_data_uri = f"data:image/jpeg;base64,{image_base64}"
            
            # Generate detailed description using Qwen Vision
            description = self._analyze_image_with_qwen(image_data_uri)
            
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


    
    def _analyze_image_with_qwen(self, image_url: str) -> Optional[str]:
        """Analyze image using Qwen2.5-VL-7B-Instruct via HuggingFace router"""
        try:
            result = requests.post(
                f"https://router.huggingface.co/fireworks-ai/inference/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe this product in detail based on the image. "
                                            "Mention product type, color, style, material, visible brand, text or labels, and key features. "
                                            "Keep it precise and structured."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url
                                    }
                                }
                            ]
                        }
                    ],
                    "model": "accounts/fireworks/models/llama4-maverick-instruct-basic"
                },
                timeout=30
            )

            print(result.json())

            if result.status_code == 200:
                return result.json()["choices"][0]["message"]["content"]
            else:
                print(f"Qwen router error: {result.status_code} - {result}")
                return None

        except Exception as e:
            print(f"Router-based Qwen vision analysis error: {str(e)}")
            return None

    
    def extract_features_from_description(self, description: str) -> dict:
        """Extract structured features from image description"""
        try:
            # Use Qwen to extract structured information
            # feature_prompt = f"""
            # From this product description, extract the following information in JSON format:
            # {{
            #     "product_type": "category of product",
            #     "color": "main color",
            #     "style": "style/design",
            #     "material": "material if mentioned",
            #     "brand": "brand if visible",
            #     "key_features": ["feature1", "feature2", "feature3"]
            # }}
            
            # Description: {description}
            # """
            
            url = "https://router.huggingface.co/fireworks-ai/inference/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""
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
                            },
                        ]
                    }
                ],
                "model": "accounts/fireworks/models/llama4-maverick-instruct-basic"
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                generated_text = ""
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result["choices"][0]["message"]["content"]
                elif isinstance(result, dict):
                    generated_text = result["choices"][0]["message"]["content"]
                
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