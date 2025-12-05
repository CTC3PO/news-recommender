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
