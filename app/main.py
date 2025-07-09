from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
import time
from .services.agents.crew_manager import CrewManager

app = FastAPI(
    title="Product Recommendation",
    description="Multi-agent system for product recommendations using text and image inputs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize crew manager
crew_manager = CrewManager()

@app.post("/recommend")
async def recommend_products(
    prompt: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    top_k: int = Form(5)
):
    """
    Get product recommendations based on text prompt and/or image
    
    - **prompt**: Text description of what you're looking for
    - **image**: Image file to analyze (optional)
    - **top_k**: Number of recommendations to return (default: 5)
    """
    start_time = time.time()
    
    try:
        # Validate inputs
        if not prompt and not image:
            raise HTTPException(
                status_code=400, 
                detail="Either prompt or image must be provided"
            )
        
        # Validate top_k
        if top_k < 1 or top_k > 20:
            raise HTTPException(
                status_code=400,
                detail="top_k must be between 1 and 20"
            )
        
        # Process image if provided
        image_bytes = None
        if image:
            # Validate image file
            if not image.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file must be an image"
                )
            
            # Check file size (max 10MB)
            contents = await image.read()
            if len(contents) > 10 * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail="Image file too large (max 10MB)"
                )
            
            image_bytes = contents
        
        # Process request
        result = crew_manager.process_request(
            user_input=prompt,
            image_bytes=image_bytes,
            top_k=top_k
        )
        
        # Add processing time
        processing_time = time.time() - start_time
        result['processing_time'] = round(processing_time, 2)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Product Recommendation Crew",
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Product Recommendation Crew API",
        "version": "1.0.0",
        "endpoints": {
            "recommend": "/recommend",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)