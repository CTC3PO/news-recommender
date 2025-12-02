# News Recommendation System

ML-powered personalized news recommender using embeddings and ranking.

## Features
- Semantic search with sentence transformers
- ML-based ranking
- Personalized recommendations
- Real-time inference

## Tech Stack
- SentenceTransformers for embeddings
- FAISS for vector search
- Scikit-learn for ranking
- FastAPI + Streamlit

## Quick Start

\`\`\bash
# Install dependencies
pip install -r requirements.txt

# Download and process data
python data/download_data.py
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