from crewai import Agent
from typing import List, Dict, Optional
import numpy as np
from ..recommend import Recommender

class ProductAgent:
    def __init__(self):
        self.recommender = Recommender()
        self.agent = Agent(
            role="E-commerce Product Specialist",
            goal="Find the most relevant products for users using advanced search techniques",
            backstory=(
                "Expert in product recommendations with deep knowledge of "
                "e-commerce products and customer preferences. Uses both text "
                "and image analysis to provide personalized recommendations."
            ),
            allow_delegation=False,
            verbose=True
        )
    
    def get_recommendations(
        self, 
        text_query: Optional[str] = None,
        image_vector: Optional[np.ndarray] = None,
        image_description: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """Get product recommendations using various inputs"""
        try:
            # Combine text query with image description for better search
            combined_query = self._combine_queries(text_query, image_description)
            
            # Use hybrid search with combined query and image vector
            results = self.recommender.hybrid_search(
                text=combined_query,
                image_vector=image_vector,
                top_k=top_k
            )
            
            # Enhance results with relevance scoring
            enhanced_results = self._enhance_results(results, text_query, image_description)
            
            return enhanced_results
            
        except Exception as e:
            print(f"Recommendation error: {str(e)}")
            return []
    
    def _combine_queries(self, text_query: Optional[str], image_description: Optional[str]) -> Optional[str]:
        """Combine text query with image description"""
        if text_query and image_description:
            return f"{text_query}. Product appears to be: {image_description}"
        elif text_query:
            return text_query
        elif image_description:
            return image_description
        else:
            return None
    
    def _enhance_results(self, results: List[Dict], text_query: Optional[str], image_description: Optional[str]) -> List[Dict]:
        """Enhance search results with additional relevance information"""
        enhanced = []
        for result in results:
            # Add relevance score based on multiple factors
            relevance_score = self._calculate_relevance(result, text_query, image_description)
            result['relevance_score'] = relevance_score
            
            # Add recommendation reason
            result['recommendation_reason'] = self._generate_recommendation_reason(
                result, text_query, image_description
            )
            
            enhanced.append(result)
        
        # Sort by relevance score
        enhanced.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return enhanced
    
    def _calculate_relevance(self, product: Dict, text_query: Optional[str], image_description: Optional[str]) -> float:
        """Calculate relevance score for a product"""
        score = 0.5  # Base score
        
        # Boost score based on product attributes
        if product.get('rating'):
            try:
                rating = float(str(product['rating']).replace('★', '').strip())
                score += (rating / 5.0) * 0.2  # Max 0.2 boost for rating
            except:
                pass
        
        # Boost for discount
        if product.get('discount_price') and product.get('actual_price'):
            try:
                discount_price = float(str(product['discount_price']).replace('₹', '').replace(',', ''))
                actual_price = float(str(product['actual_price']).replace('₹', '').replace(',', ''))
                if discount_price < actual_price:
                    score += 0.1  # Boost for discounted items
            except:
                pass
        
        return min(score, 1.0)
    
    def _generate_recommendation_reason(self, product: Dict, text_query: Optional[str], image_description: Optional[str]) -> str:
        """Generate reason for recommendation"""
        reasons = []
        
        if text_query and any(word in str(product.get('name', '')).lower() for word in text_query.lower().split()):
            reasons.append("Matches your search query")
        
        if image_description:
            reasons.append("Similar to uploaded image")
        
        if product.get('rating'):
            try:
                rating = float(str(product['rating']).replace('★', '').strip())
                if rating >= 4.0:
                    reasons.append(f"High rating ({rating}★)")
            except:
                pass
        
        if product.get('discount_price') and product.get('actual_price'):
            try:
                discount_price = float(str(product['discount_price']).replace('₹', '').replace(',', ''))
                actual_price = float(str(product['actual_price']).replace('₹', '').replace(',', ''))
                if discount_price < actual_price:
                    discount_pct = round((1 - discount_price/actual_price) * 100)
                    reasons.append(f"On sale ({discount_pct}% off)")
            except:
                pass
        
        return "; ".join(reasons) if reasons else "Recommended for you" 