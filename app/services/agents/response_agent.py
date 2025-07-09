from crewai import Agent
from typing import List, Dict, Optional
import json

class ResponseAgent:
    def __init__(self):
        self.agent = Agent(
            role="Customer Response Specialist",
            goal="Generate helpful, personalized product responses",
            backstory=(
                "Creates engaging, natural responses about product "
                "recommendations with excellent communication skills. "
                "Understands customer needs and provides detailed explanations."
            ),
            allow_delegation=False,
            verbose=True
        )
    
    def generate_response(
        self, 
        products: List[Dict], 
        user_prompt: Optional[str] = None,
        image_description: Optional[str] = None
    ) -> str:
        """Generate comprehensive response based on recommendations"""
        try:
            if not products:
                return self._generate_no_results_response(user_prompt, image_description)
            
            # Generate contextual response
            response_parts = []
            
            # Add greeting and context
            response_parts.append(self._generate_greeting(user_prompt, image_description))
            
            # Add product recommendations
            response_parts.append(self._generate_product_list(products[:3]))
            
            # Add additional insights
            response_parts.append(self._generate_insights(products))
            
            # Add closing
            response_parts.append("\nWould you like more details about any of these products or see additional recommendations?")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            print(f"Response generation error: {str(e)}")
            return "I encountered an error while preparing your recommendations. Please try again."
    
    def _generate_greeting(self, user_prompt: Optional[str], image_description: Optional[str]) -> str:
        """Generate contextual greeting"""
        if user_prompt and image_description:
            return f"Based on your request '{user_prompt}' and the uploaded image, here are my recommendations:"
        elif user_prompt:
            return f"Based on your search for '{user_prompt}', here are some great options:"
        elif image_description:
            return "Based on the image you uploaded, here are similar products I found:"
        else:
            return "Here are some product recommendations for you:"
    
    def _generate_product_list(self, products: List[Dict]) -> str:
        """Generate formatted product list"""
        product_lines = []
        
        for i, product in enumerate(products, 1):
            name = product.get('name', 'Unknown Product')
            price = product.get('discount_price', product.get('actual_price', 'N/A'))
            rating = product.get('rating', 'N/A')
            reason = product.get('recommendation_reason', '')
            
            # Format price
            if price and price != 'N/A':
                price_str = f"₹{price}" if not str(price).startswith('₹') else str(price)
            else:
                price_str = "Price not available"
            
            # Format rating
            if rating and rating != 'N/A':
                rating_str = f"({rating})" if not str(rating).endswith('★') else f"({rating})"
            else:
                rating_str = ""
            
            line = f"{i}. **{name}** - {price_str} {rating_str}"
            if reason:
                line += f"\n   *{reason}*"
            
            product_lines.append(line)
        
        return "\n".join(product_lines)
    
    def _generate_insights(self, products: List[Dict]) -> str:
        """Generate additional insights about the recommendations"""
        insights = []
        
        # Price range insight
        prices = []
        for product in products:
            price_str = product.get('discount_price', product.get('actual_price', ''))
            if price_str:
                try:
                    price = float(str(price_str).replace('₹', '').replace(',', ''))
                    prices.append(price)
                except:
                    pass
        
        if prices:
            min_price = min(prices)
            max_price = max(prices)
            if min_price != max_price:
                insights.append(f"Price range: ₹{min_price:,.0f} - ₹{max_price:,.0f}")
        
        # Discount insight
        discounted_count = sum(1 for p in products if p.get('discount_price') and p.get('actual_price'))
        if discounted_count > 0:
            insights.append(f"{discounted_count} items are currently on sale")
        
        # Rating insight
        high_rated = sum(1 for p in products if self._get_rating_value(p.get('rating', '')) >= 4.0)
        if high_rated > 0:
            insights.append(f"{high_rated} items have excellent ratings (4+ stars)")
        
        if insights:
            return f"\n**Quick Insights:** {' | '.join(insights)}"
        return ""
    
    def _get_rating_value(self, rating_str: str) -> float:
        """Extract numeric rating value"""
        try:
            return float(str(rating_str).replace('★', '').strip())
        except:
            return 0.0
    
    def _generate_no_results_response(self, user_prompt: Optional[str], image_description: Optional[str]) -> str:
        """Generate response when no products are found"""
        if user_prompt and image_description:
            return f"I couldn't find exact matches for '{user_prompt}' and the uploaded image. Try adjusting your search terms or uploading a different image."
        elif user_prompt:
            return f"I couldn't find products matching '{user_prompt}'. Try using different keywords or browse our categories."
        elif image_description:
            return "I couldn't find similar products for the uploaded image. Try uploading a clearer image or add a text description."
        else:
            return "I couldn't find any products. Please provide a search query or upload an image."
