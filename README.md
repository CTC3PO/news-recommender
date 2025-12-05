# News Recommendation System

ML-powered personalized news recommender using embeddings and ranking.

## Project Overview: Real-Time News Recommendation Engine
Build a production-grade system that recommends personalized news articles using embeddings, retrieval, and LLMs, with real-time inference and A/B testing capabilities.

## Features
- Semantic search with sentence transformers
- ML-based ranking
- Personalized recommendations
- Real-time inference
  
## System Architecture ##

- Data Pipeline (Batch + Streaming)
- Embedding & Retrieval System
- LLM-Powered Ranking & Personalization
Model Serving Infrastructure
Monitoring & Experimentation Framework

## Technical Implementation ##
### 1. Data Pipeline ###
Dataset: MIND (Microsoft News Dataset) from Kaggle

160k+ news articles with user interaction history
Rich metadata (categories, entities, abstracts)

**Tools:**

- Batch processing: Apache Spark (PySpark) for historical data
- Streaming: Kafka for real-time user events
- Storage: Feature store (Feast) + Vector DB (Weaviate/Milvus)

### 2. Embedding & Retrieval System ###
Models from HuggingFace:

- sentence-transformers/all-MiniLM-L6-v2 - Fast article embeddings
- BAAI/bge-large-en-v1.5 - High-quality retrieval embeddings

### 3. LLM-Powered Ranking ###
Models:

Base ranker: Fine-tune distilbert-base-uncased on MIND click data
LLM re-ranker: Use meta-llama/Llama-3.2-1B-Instruct for contextual ranking

### 4. Model Serving Infrastructure ###
Tech Stack:

- API: FastAPI
- Model Serving:

   -  TorchServe for PyTorch models
   -  vLLM for LLM inference optimization


- Caching: Redis for frequently accessed embeddings
- Load Balancing: Nginx

## Quick Start

\`\`\bash
# Install dependencies
pip install -r requirements.txt

# Download and process data
download data from: https://www.kaggle.com/datasets/arashnic/mind-news-dataset

python data/feature_engineering.py

# Train models
python models/embeddings.py

python models/train_ranker.py

# Run application
python app/api.py  # Terminal 1

streamlit run app/streamlit_app.py  # Terminal 2
\`\`\`

## Dataset
Microsoft MIND (News) Dataset - 50K+ articles, 50K user behaviors

##  **User Interaction Flow**
```
START
  ↓
┌─────────────────────────┐
│ User opens Streamlit app │
│ http://localhost:8501   │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Landing page shows:     │
│ - How it works          │
│ - Tech stack            │
│ - Dataset stats         │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ User chooses mode:      │
│ ○ Search                │
│ ○ Personalized          │
└──────────┬──────────────┘
           ↓
     ┌─────┴─────┐
     │           │
     ↓           ↓
┌─────────┐ ┌──────────┐
│ Search  │ │Personalized│
│ Mode    │ │  Mode    │
└────┬────┘ └────┬─────┘
     │           │
     ↓           ↓
Enter query  Select user
     │           │
     └─────┬─────┘
           ↓
┌─────────────────────────┐
│ Adjust number of results│
│ (slider: 5-20)          │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Click "Get Recommendations"│
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ API call to FastAPI     │
│ /recommend endpoint     │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Display results:        │
│ - Metrics (avg score)   │
│ - Article cards         │
│   • Title               │
│   • Category            │
│   • Abstract            │
│   • Stats (CTR, views)  │
│   • ML Score            │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ User can:               │
│ - Change settings       │
│ - Get new recommendations│
│ - Switch modes          │
└─────────────────────────┘
