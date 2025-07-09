from crewai import Crew, Process
from typing import Optional, Dict, List
from .product_agent import ProductAgent
from .vision_agent import VisionAgent
from .response_agent import ResponseAgent

class CrewManager:
    def __init__(self):
        self.product_agent = ProductAgent()
        self.vision_agent = VisionAgent()
        self.response_agent = ResponseAgent()
    
    def process_request(
        self, 
        user_input: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
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
            
            if image_bytes:
                print("Processing image with Qwen Vision...")
                image_description, image_vector = self.vision_agent.process_image(image_bytes)
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
                top_k=top_k
            )
            
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
