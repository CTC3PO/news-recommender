"""
FastAPI backend for news recommendation system
Provides REST API for personalized recommendations
"""
from models.train_ranker import SimpleRanker
from models.embeddings import EmbeddingEngine
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


# Initialize FastAPI app
app = FastAPI(
    title="News Recommender API",
    description="ML-powered personalized news recommendation system",
    version="1.0.0"
)

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and data
embedding_engine = None
ranker = None
news_df = None
user_df = None

# Pydantic models


class RecommendRequest(BaseModel):
    user_id: Optional[str] = Field(
        None, description="User ID for personalized recommendations")
    query: Optional[str] = Field(
        None, description="Search query for content-based recommendations")
    k: int = Field(10, ge=1, le=50,
                   description="Number of recommendations to return")


class Article(BaseModel):
    news_id: str
    title: str
    category: str
    subcategory: Optional[str]
    abstract: Optional[str]
    score: float
    ctr: float
    impressions: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    total_articles: int
    total_users: int


class UserInfo(BaseModel):
    user_id: str
    articles_read: int
    top_category: str
    category_diversity: float


@app.on_event("startup")
async def load_models():
    """Load all models and data on startup"""
    global embedding_engine, ranker, news_df, user_df

    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)

    try:
        # Load embedding engine
        print("\n1. Loading embedding engine...")
        embedding_engine = EmbeddingEngine()
        embedding_engine.load("data/processed")
        print("   ✓ Embeddings ready")

        # Load ranker
        print("\n2. Loading ranking model...")
        ranker = SimpleRanker()
        ranker.load("models/ranker.pkl")
        print("   ✓ Ranker ready")

        # Load data
        print("\n3. Loading datasets...")
        news_df = pd.read_parquet("data/processed/news_features.parquet")
        user_df = pd.read_parquet("data/processed/user_features.parquet")
        print(f"   ✓ Loaded {len(news_df):,} articles")
        print(f"   ✓ Loaded {len(user_df):,} users")

        print("\n" + "="*60)
        print("✅ API READY")
        print("="*60)
        print(f"\nAPI running at: http://localhost:8000")
        print(f"Docs available at: http://localhost:8000/docs")
        print("\n")

    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        raise


@app.get("/", response_model=dict)
def root():
    """Root endpoint"""
    return {
        "message": "News Recommender API",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend",
            "users": "/users",
            "user_info": "/user/{user_id}",
            "article": "/article/{news_id}"
        }
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=embedding_engine is not None and ranker is not None,
        total_articles=len(news_df) if news_df is not None else 0,
        total_users=len(user_df) if user_df is not None else 0
    )


@app.post("/recommend", response_model=List[Article])
def get_recommendations(request: RecommendRequest):
    """
    Get personalized recommendations

    Two modes:
    1. Personalized (user_id): Uses user history + ML ranking
    2. Search (query): Content-based retrieval
    """
    try:
        # Validate input
        if not request.user_id and not request.query:
            raise HTTPException(
                status_code=400,
                detail="Must provide either user_id or query"
            )

        # Step 1: Retrieve candidates using embeddings
        if request.query:
            # Content-based: use query embedding
            query_emb = embedding_engine.model.encode(request.query)
            mode = "search"
        else:
            # Personalized: use user's top category as query
            user_profile = user_df[user_df['user_id'] == request.user_id]

            if len(user_profile) == 0:
                raise HTTPException(status_code=404, detail="User not found")

            top_category = user_profile.iloc[0]['top_category']
            query_text = f"{top_category} news"
            query_emb = embedding_engine.model.encode(query_text)
            mode = "personalized"

        # Retrieve more candidates than needed for re-ranking
        candidates = embedding_engine.search(query_emb, k=request.k * 3)

        # Get full article information
        candidate_ids = [c['news_id'] for c in candidates]
        candidate_articles = news_df[news_df['news_id'].isin(
            candidate_ids)].copy()

        # Step 2: Re-rank with ML model (if personalized)
        if mode == "personalized" and len(candidate_articles) > 0:
            user_profile = user_df[user_df['user_id']
                                   == request.user_id].iloc[0]

            # Extract features for all candidates
            features = []
            for _, article in candidate_articles.iterrows():
                feat = ranker._extract_features(user_profile, article)
                features.append(feat)

            features_df = pd.DataFrame(features)
            scores = ranker.predict(features_df)

            candidate_articles['ml_score'] = scores
            candidate_articles = candidate_articles.sort_values(
                'ml_score', ascending=False)
        else:
            # For search mode, use embedding similarity as score
            candidate_articles['ml_score'] = 0.5

        # Step 3: Format and return top-k
        results = []
        for _, row in candidate_articles.head(request.k).iterrows():
            results.append(Article(
                news_id=row['news_id'],
                title=row['title'],
                category=row['category'],
                subcategory=row.get('subcategory'),
                abstract=row.get('abstract') or "",
                score=float(row['ml_score']),
                ctr=float(row.get('ctr', 0)),
                impressions=int(row.get('impressions', 0))
            ))

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating recommendations: {str(e)}")


@app.get("/users", response_model=dict)
def list_users(limit: int = Query(20, ge=1, le=100)):
    """Get list of sample user IDs"""
    sample_users = user_df['user_id'].head(limit).tolist()
    return {
        "users": sample_users,
        "total_users": len(user_df),
        "returned": len(sample_users)
    }


@app.get("/user/{user_id}", response_model=UserInfo)
def get_user_info(user_id: str):
    """Get user profile information"""
    user = user_df[user_df['user_id'] == user_id]

    if len(user) == 0:
        raise HTTPException(status_code=404, detail="User not found")

    user = user.iloc[0]

    return UserInfo(
        user_id=user['user_id'],
        articles_read=int(user['num_articles_read']),
        top_category=user['top_category'],
        category_diversity=float(user['category_diversity'])
    )


@app.get("/article/{news_id}", response_model=Article)
def get_article(news_id: str):
    """Get article details"""
    article = news_df[news_df['news_id'] == news_id]

    if len(article) == 0:
        raise HTTPException(status_code=404, detail="Article not found")

    article = article.iloc[0]

    return Article(
        news_id=article['news_id'],
        title=article['title'],
        category=article['category'],
        subcategory=article.get('subcategory'),
        abstract=article.get('abstract') or "",
        score=0.0,
        ctr=float(article.get('ctr', 0)),
        impressions=int(article.get('impressions', 0))
    )


@app.get("/categories", response_model=dict)
def list_categories():
    """Get all categories with article counts"""
    category_counts = news_df['category'].value_counts().to_dict()
    return {
        "categories": category_counts,
        "total": len(category_counts)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
