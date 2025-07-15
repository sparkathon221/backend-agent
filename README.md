# 🛍️ AI Shopping Assistant Backend 
A FastAPI-based backend service powering an AI shopping assistant using: 
- ✅ FAISS vector search over product data 
- 🤖 CrewAI agents for reasoning 
- ⚡ Easily pluggable with a Next.js frontend 

--- 
## 🚀 Quickstart 
### 1. Clone the repository 
```bash 
git clone git@github.com:sparkathon221/backend-agent.git 
cd backend-agent 
``` 
### 2. Setup virtual environment 
```bash 
python3 -m venv .venv 
source .venv/bin/activate 
``` 
### 3. Install dependencies 
```bash 
pip install -r requirements.txt 
``` 

--- 
## 🧱 Dataset Preparation 
### Step 1: Extract and preprocess dataset 
```bash 
./load_dataset.sh 
python3 scripts/parse_data.py 
``` 
This will: 
- Unzip and merge CSVs 
- Keep top 1000 rows per file 
- Assign unique ProductIDs 
### Step 2: Build FAISS vector index 
```bash 
python3 scripts/build_faiss.py 
``` 
This will: 
- Load the merged dataset 
- Generate embeddings 
- Create and store a FAISS index 

--- 
## 📡 Run the API Server 
```bash 
uvicorn app.main:app --reload 
``` 
Access your endpoints at: http://localhost:8000 

--- 
## 🧠 Agent Logic 
The backend includes: 
- 🧾 Product retriever with FAISS 
- 🧠 CrewAI-based reasoning agent 
- 🔌 Endpoints for frontend to query product-related intents 

---
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
---
https://github.com/user-attachments/assets/e096a1df-19f6-4191-b4f9-fdc9b49cd1a8
--- 
