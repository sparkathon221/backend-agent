from crewai import Crew, Process
from typing import Optional, Dict, List
from .product_agent import ProductAgent
from .vision_agent import VisionAgent
from .response_agent import ResponseAgent
from concurrent.futures import ThreadPoolExecutor

class CrewManager:
    def __init__(self):
        self.product_agent = ProductAgent()
        self.vision_agent = VisionAgent()
        self.response_agent = ResponseAgent()
    
    def process_request(
        self, 
        user_input: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_url: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, any]:
        """Process user request and return comprehensive recommendations"""
        try:
            # Initialize context
            context = {
                'user_input': user_input or "",
                'has_image': image_bytes is not None,
                'top_k': top_k
            }
            
            # Process image if provided
            image_description = None
            image_vector = None
            image_features = {}
            image_descriptions = {}
            
            if image_bytes:
                print("Processing image with Qwen Vision...")
                image_description, image_vector = self.vision_agent.process_image(image_bytes)
                if image_description:
                    image_descriptions['user_image'] = image_description
                context['image_description'] = image_description or ""
                
                # Extract structured features from description
                if image_description:
                    image_features = self.vision_agent.extract_features_from_description(image_description)
                    context['image_features'] = image_features
            elif image_url:
                print(f"Processing image from URL: {image_url}")
                image_description, image_vector = self.vision_agent.process_image_url(image_url)
                if image_description:
                    image_descriptions['user_image'] = image_description
                context['image_description'] = image_description or ""
                
                # Extract structured features from description
                if image_description:
                    image_features = self.vision_agent.extract_features_from_description(image_description)
                    context['image_features'] = image_features
            
            # Get product recommendations
            print("Getting product recommendations...")
            recommendations = self.product_agent.get_recommendations(
                text_query=user_input,
                image_vector=image_vector,
                image_description=image_description,
                top_k=top_k * 2  # Get more initial results for better filtering
            )
            
            # Process product images in parallel
            if recommendations:
                print("Processing product images...")
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for product in recommendations:
                        if 'image_url' in product:
                            future = executor.submit(
                                self.vision_agent.process_image_url,
                                product['image_url']
                            )
                            futures.append((product, future))
                    
                    # Collect image descriptions
                    for product, future in futures:
                        try:
                            description, _ = future.result()
                            if description:
                                image_descriptions[product['image_url']] = description
                        except Exception as e:
                            print(f"Error processing product image {product.get('product_id', 'unknown')}: {str(e)}")
            
            # Sort recommendations by relevance using both text and image information
            if recommendations and user_input and image_descriptions:
                print("Sorting recommendations by relevance...")
                recommendations = self.vision_agent.sort_products_by_relevance(
                    recommendations,
                    user_input,
                    image_descriptions
                )
                
                # Take top_k results after sorting
                recommendations = recommendations[:top_k]
            
            context['recommendations'] = recommendations
            
            # Generate response
            print("Generating response...")
            response_text = self.response_agent.generate_response(
                products=recommendations,
                user_prompt=user_input,
                image_description=image_description
            )
            
            # Prepare final result
            result = {
                "recommendations": recommendations,
                "response": response_text,
                "context": {
                    "user_input": user_input,
                    "image_description": image_description,
                    "image_features": image_features,
                    "total_results": len(recommendations)
                }
            }
            
            return result
            
        except Exception as e:
            print(f"CrewManager error: {str(e)}")
            return {
                "recommendations": [],
                "response": f"I encountered an error while processing your request: {str(e)}",
                "context": {"error": str(e)}
            }
