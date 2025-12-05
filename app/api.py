"""
FastAPI backend for news recommendation system
Provides REST API for personalized recommendations
"""
from typing import List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import json
from fastapi import Depends

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


# Try to import models with error handling
try:
    from models.train_ranker import SimpleRanker
    from models.embeddings import EmbeddingEngine
    MODELS_LOADED = True
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    print("API will run in limited mode (static data only)")
    MODELS_LOADED = False
    # Define dummy classes if imports fail

    class SimpleRanker:
        def __init__(self): pass
        def load(self, path): pass
        def predict(self, features_df): return np.zeros(len(features_df))

    class EmbeddingEngine:
        def __init__(self): pass
        def load(self, path): pass
        def search(self, query_emb, k): return []


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
        # Load data
        print("\n1. Loading datasets...")
        news_path = Path("data/processed/news_features.parquet")
        user_path = Path("data/processed/user_features.parquet")

        if not news_path.exists():
            print(f"   ❌ News features not found: {news_path}")
            # Load raw data as fallback
            news_df = load_raw_news()
        else:
            news_df = pd.read_parquet(news_path)
            print(f"   ✓ Loaded {len(news_df):,} articles")

        if not user_path.exists():
            print(f"   ❌ User features not found: {user_path}")
            user_df = pd.DataFrame(columns=['user_id'])
        else:
            user_df = pd.read_parquet(user_path)
            print(f"   ✓ Loaded {len(user_df):,} users")

        # Load embedding engine if available
        if MODELS_LOADED:
            print("\n2. Loading embedding engine...")
            embedding_engine = EmbeddingEngine()
            # Check if embeddings exist
            embedding_path = Path("data/processed/article_embeddings.npy")
            if embedding_path.exists():
                embedding_engine.load("data/processed")
                print("   ✓ Embeddings ready")
            else:
                print("   ⚠️  Embeddings not found, using fallback")
                embedding_engine = None

        # Load ranker if available
        if MODELS_LOADED:
            print("\n3. Loading ranking model...")
            ranker_path = Path("models/ranker.pkl")
            if ranker_path.exists():
                ranker = SimpleRanker()
                ranker.load("models/ranker.pkl")
                print("   ✓ Ranker ready")
            else:
                print("   ⚠️  Ranker model not found, using popularity ranking")
                ranker = None

        print("\n" + "="*60)
        print("✅ API READY")
        print("="*60)
        print(f"\nAPI running at: http://localhost:8000")
        print(f"Docs available at: http://localhost:8000/docs")
        print("\n")

    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        # Continue anyway with limited functionality
        print("⚠️  Starting API with limited functionality")


def load_raw_news():
    """Load raw news data if processed data not available"""
    try:
        news_path = Path("data/raw/train/news.tsv")
        if news_path.exists():
            columns = [
                'news_id', 'category', 'subcategory', 'title',
                'abstract', 'url', 'title_entities', 'abstract_entities'
            ]
            df = pd.read_csv(news_path, sep='\t', header=None,
                             names=columns, on_bad_lines='warn')
            df = df[['news_id', 'category', 'subcategory', 'title', 'abstract']]
            # Add dummy features
            df['ctr'] = 0.0
            df['impressions'] = 0
            return df
    except Exception:
        pass

    # Return empty dataframe as last resort
    return pd.DataFrame(columns=['news_id', 'category', 'title', 'abstract', 'ctr', 'impressions'])


@app.get("/", response_model=dict)
def root():
    """Root endpoint"""
    return {
        "message": "News Recommender API",
        "status": "running",
        "models_available": MODELS_LOADED,
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
        # Validate we have data
        if news_df is None or len(news_df) == 0:
            raise HTTPException(
                status_code=503, detail="No articles available")

        # Validate input
        if not request.user_id and not request.query:
            raise HTTPException(
                status_code=400,
                detail="Must provide either user_id or query"
            )

        # If no models available, use simple popularity ranking
        if embedding_engine is None or ranker is None:
            return get_popular_recommendations(request)

        # Step 1: Retrieve candidates using embeddings
        if request.query:
            # Content-based: use query embedding
            query_emb = embedding_engine.model.encode(request.query)
            mode = "search"
        else:
            # Personalized: use user's top category as query
            if user_df is None or request.user_id not in user_df['user_id'].values:
                # Return popular articles for unknown users
                return get_popular_recommendations(request)

            user_profile = user_df[user_df['user_id']
                                   == request.user_id].iloc[0]
            top_category = user_profile['top_category']
            query_text = f"{top_category} news"
            query_emb = embedding_engine.model.encode(query_text)
            mode = "personalized"

        # Retrieve candidates
        candidates = embedding_engine.search(query_emb, k=request.k * 3)

        # Get full article information
        candidate_ids = [c['news_id'] for c in candidates]
        candidate_articles = news_df[news_df['news_id'].isin(
            candidate_ids)].copy()

        # Step 2: Re-rank with ML model (if personalized)
        if mode == "personalized" and ranker is not None and len(candidate_articles) > 0:
            # Extract features for all candidates
            features = []
            for _, article in candidate_articles.iterrows():
                feat = ranker._extract_features(user_profile, article)
                features.append(feat)

            if features:
                features_df = pd.DataFrame(features)
                scores = ranker.predict(features_df)
                candidate_articles['ml_score'] = scores
                candidate_articles = candidate_articles.sort_values(
                    'ml_score', ascending=False)
            else:
                candidate_articles['ml_score'] = 0.5
        else:
            # For search mode or no ranker, use embedding similarity as score
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


def get_popular_recommendations(request: RecommendRequest):
    """Fallback: return popular articles sorted by impressions"""
    if news_df is None or len(news_df) == 0:
        raise HTTPException(status_code=503, detail="No articles available")

    # Sort by impressions (or use random if no impressions)
    if 'impressions' in news_df.columns:
        sorted_news = news_df.sort_values('impressions', ascending=False)
    else:
        sorted_news = news_df.sample(frac=1, random_state=42)

    results = []
    for _, row in sorted_news.head(request.k).iterrows():
        results.append(Article(
            news_id=row['news_id'],
            title=row['title'],
            category=row['category'],
            subcategory=row.get('subcategory'),
            abstract=row.get('abstract') or "",
            score=0.0,
            ctr=float(row.get('ctr', 0)),
            impressions=int(row.get('impressions', 0))
        ))

    return results


@app.get("/users", response_model=dict)
def list_users(limit: int = Query(20, ge=1, le=100)):
    """Get list of sample user IDs"""
    if user_df is None or len(user_df) == 0:
        return {"users": [], "total_users": 0, "returned": 0}

    sample_users = user_df['user_id'].head(limit).tolist()
    return {
        "users": sample_users,
        "total_users": len(user_df),
        "returned": len(sample_users)
    }


@app.get("/user/{user_id}", response_model=UserInfo)
def get_user_info(user_id: str):
    """Get user profile information"""
    if user_df is None or len(user_df) == 0:
        raise HTTPException(status_code=404, detail="No user data available")

    user = user_df[user_df['user_id'] == user_id]

    if len(user) == 0:
        raise HTTPException(status_code=404, detail="User not found")

    user = user.iloc[0]

    return UserInfo(
        user_id=user['user_id'],
        articles_read=int(user.get('num_articles_read', 0)),
        top_category=user.get('top_category', 'unknown'),
        category_diversity=float(user.get('category_diversity', 0))
    )


@app.get("/article/{news_id}", response_model=Article)
def get_article(news_id: str):
    """Get article details"""
    if news_df is None or len(news_df) == 0:
        raise HTTPException(status_code=404, detail="No articles available")

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
    if news_df is None or len(news_df) == 0:
        return {"categories": {}, "total": 0}

    category_counts = news_df['category'].value_counts().to_dict()
    return {
        "categories": category_counts,
        "total": len(category_counts)
    }

# Add these to your API


class UserInteraction(BaseModel):
    user_id: str
    article_id: str
    action: str  # 'click', 'like', 'share', 'read'
    timestamp: str


@app.post("/log_interaction")
def log_interaction(interaction: UserInteraction):
    """Log user interactions for real-time learning"""
    # Store in database (Redis/SQLite for demo)
    # Update user profile in real-time
    # Return updated recommendations


@app.get("/trending_now")
def get_trending_now(limit: int = 10):
    """Get currently trending articles based on recent clicks"""
    # Calculate trending based on last hour/day
    # Real-time popularity ranking


class UserInteraction(BaseModel):
    user_id: str
    article_id: str
    action: str  # 'click', 'like', 'dislike', 'read', 'share'
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    duration: Optional[float] = None  # Reading time in seconds


class TrendingRequest(BaseModel):
    time_window: str = Field("1h", description="Time window: 1h, 24h, 7d")
    limit: int = Field(10, ge=1, le=50)


# Add global variables for real-time tracking
user_interactions = []  # In production, use Redis or database
trending_cache = {}
last_trending_update = None

# Add these endpoints AFTER your existing endpoints


@app.post("/log_interaction")
async def log_interaction(interaction: UserInteraction):
    """
    Log user interaction for real-time learning
    """
    global user_interactions

    # Store interaction
    user_interactions.append(interaction.dict())

    # Keep only last 10,000 interactions (for demo)
    if len(user_interactions) > 10000:
        user_interactions = user_interactions[-10000:]

    # In production: Update user profile in real-time
    # update_user_profile(interaction.user_id, interaction)

    return {
        "status": "logged",
        "interaction_id": len(user_interactions),
        "message": "Interaction recorded successfully"
    }


@app.get("/trending", response_model=List[Article])
def get_trending(request: TrendingRequest = Depends()):
    """
    Get trending articles based on recent interactions
    """
    global trending_cache, last_trending_update

    # Cache trending for 5 minutes
    cache_key = f"{request.time_window}_{request.limit}"
    current_time = datetime.now()

    if (last_trending_update and
        (current_time - last_trending_update).seconds < 300 and
            cache_key in trending_cache):
        return trending_cache[cache_key]

    # Calculate time window
    if request.time_window == "1h":
        time_delta = timedelta(hours=1)
    elif request.time_window == "24h":
        time_delta = timedelta(days=1)
    else:  # 7d
        time_delta = timedelta(days=7)

    cutoff_time = current_time - time_delta

    # Filter recent interactions
    recent_interactions = [
        i for i in user_interactions
        if datetime.fromisoformat(i['timestamp']) > cutoff_time
    ]

    # Calculate trending scores
    article_scores = {}
    for interaction in recent_interactions:
        article_id = interaction['article_id']
        if article_id not in article_scores:
            article_scores[article_id] = {
                'clicks': 0,
                'likes': 0,
                'reads': 0,
                'total_score': 0,
                'last_interaction': interaction['timestamp']
            }

        # Weight different actions
        weights = {
            'click': 1,
            'like': 3,
            'read': 2,
            'share': 5
        }

        article_scores[article_id]['total_score'] += weights.get(
            interaction['action'], 1)
        article_scores[article_id][f"{interaction['action']}s"] += 1

    # Get article details and calculate trending score
    trending_articles = []
    for article_id, scores in sorted(
        article_scores.items(),
        key=lambda x: x[1]['total_score'],
        reverse=True
    )[:request.limit]:

        # Get article info
        article_info = news_df[news_df['news_id'] == article_id]
        if len(article_info) > 0:
            article = article_info.iloc[0]

            # Calculate velocity (recent popularity)
            recent_hour = datetime.now() - timedelta(hours=1)
            recent_interactions_count = len([
                i for i in recent_interactions
                if i['article_id'] == article_id and
                datetime.fromisoformat(i['timestamp']) > recent_hour
            ])

            trending_score = scores['total_score'] * \
                (1 + recent_interactions_count * 0.1)

            trending_articles.append(Article(
                news_id=article_id,
                title=article['title'],
                category=article['category'],
                subcategory=article.get('subcategory'),
                abstract=article.get('abstract') or "",
                score=float(trending_score / 100),  # Normalize
                ctr=float(article.get('ctr', 0)),
                impressions=int(article.get('impressions', 0))
            ))

    # Cache results
    trending_cache[cache_key] = trending_articles
    last_trending_update = current_time

    return trending_articles


@app.get("/user/{user_id}/interactions")
def get_user_interactions(user_id: str, limit: int = 20):
    """
    Get user's interaction history
    """
    user_ints = [
        i for i in user_interactions
        if i['user_id'] == user_id
    ][-limit:]  # Get most recent

    return {
        "user_id": user_id,
        "total_interactions": len(user_ints),
        "interactions": user_ints
    }


@app.get("/article/{news_id}/stats")
def get_article_stats(news_id: str):
    """
    Get real-time statistics for an article
    """
    article_ints = [i for i in user_interactions if i['article_id'] == news_id]

    # Calculate stats
    stats = {
        "total_interactions": len(article_ints),
        "clicks": sum(1 for i in article_ints if i['action'] == 'click'),
        "likes": sum(1 for i in article_ints if i['action'] == 'like'),
        "reads": sum(1 for i in article_ints if i['action'] == 'read'),
        "shares": sum(1 for i in article_ints if i['action'] == 'share'),
        "first_seen": min((i['timestamp'] for i in article_ints), default=None),
        "last_seen": max((i['timestamp'] for i in article_ints), default=None)
    }

    # Calculate engagement rate
    if article_ints:
        # In production, you'd use actual impression data
        stats["estimated_engagement"] = len(article_ints) / 100  # Simplified

    return stats

# Add a new recommendation mode


@app.post("/recommend/explore")
def explore_recommendations(request: RecommendRequest):
    """
    Exploration recommendations - shows diverse content
    """
    if not request.user_id:
        raise HTTPException(400, "User ID required for exploration")

    # Get user's usual categories
    user_profile = user_df[user_df['user_id'] == request.user_id]
    if len(user_profile) == 0:
        raise HTTPException(404, "User not found")

    usual_categories = [user_profile.iloc[0]['top_category']]
    if user_profile.iloc[0]['second_category']:
        usual_categories.append(user_profile.iloc[0]['second_category'])

    # Get articles NOT in user's usual categories
    diverse_articles = news_df[~news_df['category'].isin(usual_categories)]

    if len(diverse_articles) == 0:
        diverse_articles = news_df

    # Sample diverse articles
    sample = diverse_articles.sample(min(request.k, len(diverse_articles)))

    results = []
    for _, row in sample.iterrows():
        results.append(Article(
            news_id=row['news_id'],
            title=row['title'],
            category=row['category'],
            subcategory=row.get('subcategory'),
            abstract=row.get('abstract') or "",
            score=0.5,  # Exploration score
            ctr=float(row.get('ctr', 0)),
            impressions=int(row.get('impressions', 0))
        ))

    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
