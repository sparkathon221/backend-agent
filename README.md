

# 🛍️ BUYCEPS

An intelligent product recommendation system that combines computer vision and natural language processing to provide personalized shopping experiences. The system processes both text queries and product images to deliver highly relevant product suggestions.

## 🚀 Features

- **Hybrid Search**: Combines text-based and image-based search for comprehensive product matching
- **Computer Vision**: Advanced image analysis using llama4-maverick-instruct-basic for product identification
- **Multi-Agent Architecture**: Specialized agents for product analysis, vision processing, and response generation
- **Real-time Processing**: Concurrent image processing for improved performance
- **Semantic Understanding**: Uses sentence transformers for deep semantic matching
- **Relevance Scoring**: Advanced algorithms to rank products by multiple relevance factors

## 🏗️ Architecture

The system follows a multi-agent architecture with three specialized components:

### Core Agents
- **ProductAgent**: Handles product search and recommendation logic
- **VisionAgent**: Processes images using llama4-maverick-instruct-basic model
- **ResponseAgent**: Generates natural, personalized responses

### Key Components
- **CrewManager**: Orchestrates the entire recommendation pipeline
- **Recommender**: Manages FAISS vector search and hybrid recommendations
- **Data Processing**: Automated dataset merging and embedding generation

## 🛠️ Technology Stack

- **Machine Learning**: Sentence Transformers, FAISS for vector similarity search
- **Computer Vision**: llama4-maverick-instruct-basic via HuggingFace API
- **Framework**: CrewAI for multi-agent orchestration, FastAPI-based backend service
- **Data Processing**: Pandas, NumPy for data manipulation
- **Concurrency**: ThreadPoolExecutor for parallel processing
- **APIs**: HuggingFace Transformers, RESTful services

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ecommerce-ai-recommender.git
cd ecommerce-ai-recommender
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
export HF_TOKEN="your_huggingface_token"
```

4. **Prepare the dataset**
```bash
python data_processing/merge_data.py
python data_processing/build_embeddings.py
```

## 🚀 Quick Start

```python
from crew.crew_manager import CrewManager

# Initialize the system
manager = CrewManager()

# Text-based search
result = manager.process_request(
    user_input="wireless bluetooth headphones",
    top_k=5
)

# Image-based search
with open("product_image.jpg", "rb") as f:
    image_bytes = f.read()

result = manager.process_request(
    user_input="similar products",
    image_bytes=image_bytes,
    top_k=5
)

print(result["response"])
```

## 📊 Data Pipeline

### 1. Data Collection & Processing
- Merges multiple Amazon product CSV files
- Generates sequential product IDs
- Handles missing data and normalization

### 2. Embedding Generation
- Uses `all-MiniLM-L6-v2` for text embeddings
- Normalizes vectors for cosine similarity
- Builds FAISS index for efficient search

### 3. Image Processing
- Converts images to base64 for API processing
- Generates detailed product descriptions
- Creates semantic embeddings from descriptions

## 🔍 Search Capabilities

### Hybrid Search Algorithm
1. **Text Processing**: Converts queries to semantic embeddings
2. **Image Analysis**: Extracts detailed product features from images
3. **Vector Combination**: Merges text and image vectors for unified search
4. **Relevance Scoring**: Multi-factor scoring including ratings, discounts, and semantic similarity
5. **Result Ranking**: Intelligent sorting based on user intent and product attributes

### Supported Search Types
- Pure text search
- Image-only search
- Combined text + image search
- Similarity-based recommendations

## 📈 Performance Features

- **Concurrent Processing**: Parallel image analysis for multiple products
- **Efficient Vector Search**: FAISS indexing for sub-second search times
- **Memory Optimization**: Normalized embeddings for reduced storage
- **Error Handling**: Robust error management for API failures

## 🔧 Configuration

### Model Configuration
```python
# Embedding model (configurable)
MODEL_NAME = "all-MiniLM-L6-v2"

# Vision model
VISION_MODEL = "accounts/fireworks/models/llama4-maverick-instruct-basic"

# Search parameters
DEFAULT_TOP_K = 5
RELEVANCE_WEIGHTS = {
    "query_product": 0.4,
    "query_image": 0.4,
    "product_image": 0.2
}
```


## 🎯 Use Cases

- **E-commerce Platforms**: Product discovery and recommendations
- **Visual Search**: Find products similar to uploaded images
- **Cross-modal Search**: Combine text descriptions with visual similarity
- **Personalized Shopping**: Context-aware product suggestions

## 🚀 Future Enhancements

- [ ] Real-time learning from user interactions
- [ ] Advanced product categorization
- [ ] Price trend analysis integration
- [ ] Multi-language support
- [ ] Mobile app integration
- [ ] A/B testing framework for recommendation algorithms

 ## 🧪 Example Endpoints 
```bash 
GET /health 
POST /agent/query 
```
Use the `agent/query` endpoint to send user questions and receive intelligent product guidance. 

--- 

Use any cloud host or containerize with Docker: 
```bash 
uvicorn app.main:app --host 0.0.0.0 --port 8000 
```
## Demo
https://github.com/user-attachments/assets/8d91a6f0-0c99-4a98-ae92-3889d4285dd1

