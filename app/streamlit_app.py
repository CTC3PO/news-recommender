"""
Streamlit web interface for news recommendation system
Provides interactive UI for testing the recommender
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="News Recommender",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
    }
    .article-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        border-left: 4px solid #1E88E5;
    }
    .metric-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# API configuration
API_URL = "http://localhost:8000"

# Helper functions


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_recommendations(user_id=None, query=None, k=10):
    """Get recommendations from API"""
    payload = {"k": k}
    if user_id:
        payload["user_id"] = user_id
    if query:
        payload["query"] = query

    response = requests.post(f"{API_URL}/recommend", json=payload, timeout=30)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API Error: {response.status_code}")
        return None


def get_user_info(user_id):
    """Get user profile information"""
    try:
        response = requests.get(f"{API_URL}/user/{user_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def get_sample_users():
    """Get sample user IDs"""
    try:
        response = requests.get(f"{API_URL}/users", timeout=5)
        if response.status_code == 200:
            return response.json()['users']
    except:
        pass
    return []

# Main app


def main():
    # Header
    st.markdown('<p class="main-header">📰 News Recommender</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML-powered personalized content discovery</p>',
                unsafe_allow_html=True)

    # Check API health
    api_healthy = check_api_health()

    if not api_healthy:
        st.error("⚠️ Cannot connect to API. Make sure it's running:")
        st.code("python app/api.py")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Mode selection
        mode = st.radio(
            "Recommendation Mode",
            ["🔍 Search", "👤 Personalized"],
            help="Search mode uses your query, Personalized mode uses user history"
        )

        st.markdown("---")

        if mode == "👤 Personalized":
            # User selection
            sample_users = get_sample_users()

            if sample_users:
                selected_user = st.selectbox(
                    "Select User",
                    [""] + sample_users,
                    help="Choose a user to get personalized recommendations"
                )
            else:
                selected_user = st.text_input("Enter User ID:")

            # Show user profile
            if selected_user:
                with st.spinner("Loading user profile..."):
                    user_info = get_user_info(selected_user)

                    if user_info:
                        st.success("✅ User Found")
                        st.metric("Articles Read", user_info['articles_read'])
                        st.metric("Top Category", user_info['top_category'])
                        st.metric("Category Diversity",
                                  f"{user_info['category_diversity']:.2f}")

            query = None

        else:  # Search mode
            query = st.text_input(
                "🔍 Search Query",
                placeholder="e.g., artificial intelligence, sports news...",
                help="Enter keywords to find relevant articles"
            )
            selected_user = None

        st.markdown("---")

        # Number of results
        num_results = st.slider(
            "Number of Results",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )

        st.markdown("---")

        # Get recommendations button
        search_button = st.button(
            "🚀 Get Recommendations",
            type="primary",
            use_container_width=True
        )

    # Main content area
    if search_button:
        # Validate input
        if mode == "👤 Personalized" and not selected_user:
            st.warning("⚠️ Please select a user")
            return

        if mode == "🔍 Search" and not query:
            st.warning("⚠️ Please enter a search query")
            return

        # Show search info
        if mode == "👤 Personalized":
            st.subheader(f"👤 Recommendations for User: {selected_user}")
        else:
            st.subheader(f"🔍 Search Results for: '{query}'")

        # Get recommendations
        with st.spinner("🔄 Finding best articles..."):
            recommendations = get_recommendations(
                user_id=selected_user if mode == "👤 Personalized" else None,
                query=query if mode == "🔍 Search" else None,
                k=num_results
            )

        if recommendations:
            st.success(f"✅ Found {len(recommendations)} recommendations")

            # Display metrics
            col1, col2, col3 = st.columns(3)

            avg_score = sum(r['score']
                            for r in recommendations) / len(recommendations)
            avg_ctr = sum(r['ctr']
                          for r in recommendations) / len(recommendations)
            categories = len(set(r['category'] for r in recommendations))

            with col1:
                st.metric("Avg Score", f"{avg_score:.3f}")
            with col2:
                st.metric("Avg CTR", f"{avg_ctr:.3%}")
            with col3:
                st.metric("Categories", categories)

            st.markdown("---")

            # Display articles
            for i, article in enumerate(recommendations, 1):
                with st.container():
                    col1, col2 = st.columns([5, 1])

                    with col1:
                        st.markdown(f"### {i}. {article['title']}")

                        # Category badges
                        category_color = {
                            'news': '🔵',
                            'sports': '⚽',
                            'finance': '💰',
                            'entertainment': '🎬',
                            'health': '🏥',
                            'lifestyle': '✨',
                            'foodanddrink': '🍔',
                            'travel': '✈️',
                            'autos': '🚗'
                        }

                        emoji = category_color.get(
                            article['category'].lower(), '📰')
                        st.markdown(
                            f"{emoji} **{article['category']}** | *{article.get('subcategory', 'N/A')}*")

                        # Abstract
                        if article.get('abstract'):
                            st.markdown(f"_{article['abstract'][:200]}..._")

                        # Stats
                        st.caption(
                            f"👁️ {article['impressions']:,} impressions | 📊 CTR: {article['ctr']:.3%}")

                    with col2:
                        st.metric("Score", f"{article['score']: .st.markdown("---")
    else:
        st.error("❌ No recommendations found")


else:
    # Landing page
    st.markdown("""
    ## 🎯 How It Works
    
    This system uses a **two-stage recommendation pipeline**:
    
    ### 1️⃣ Candidate Retrieval
    - Uses **Sentence Transformers** to generate semantic embeddings
    - **FAISS** index enables fast similarity search
    - Retrieves top candidates based on content similarity
    
    ### 2️⃣ ML Ranking
    - **Gradient Boosting** model predicts click probability
    - Considers user preferences, article quality, and compatibility
    - Re-ranks candidates for optimal personalization
    
    ---
    
    ## 🚀 Get Started
    
    1. Choose your mode:
       - **🔍 Search**: Find articles by keywords
       - **👤 Personalized**: Get recommendations based on user history
    
    2. Configure settings in the sidebar
    
    3. Click "Get Recommendations"
    
    ---
    
    ## 🛠️ Tech Stack
    
    - **Models**: SentenceTransformers, Scikit-learn
    - **Vector DB**: FAISS (50K+ article embeddings)
    - **Backend**: FastAPI
    - **Frontend**: Streamlit
    - **Data**: Microsoft MIND News Dataset
    """)

    # Show dataset stats
    try:
        health = requests.get(f"{API_URL}/health").json()

        st.markdown("### 📊 Dataset Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Articles", f"{health['total_articles']:,}")
        with col2:
            st.metric("Total Users", f"{health['total_users']:,}")
        with col3:
            st.metric("API Status",
                      "🟢 Healthy" if health['models_loaded'] else "🔴 Error")
    except:
        pass
