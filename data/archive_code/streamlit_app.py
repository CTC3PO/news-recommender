"""
Interactive News Recommender with Real User Engagement
"""
import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import time
import random

# Page configuration
st.set_page_config(
    page_title="Interactive News Recommender",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .article-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .article-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .category-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        background: #e3f2fd;
        color: #1976d2;
    }
    .interaction-buttons {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
    }
    .interaction-btn {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
    }
    .like-btn {
        background: #4caf50;
        color: white;
    }
    .dislike-btn {
        background: #f44336;
        color: white;
    }
    .read-btn {
        background: #2196f3;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# API configuration
API_URL = "http://localhost:8000"

# Initialize session state for user interactions
if 'user_interactions' not in st.session_state:
    st.session_state.user_interactions = []
if 'article_ratings' not in st.session_state:
    st.session_state.article_ratings = {}
if 'reading_history' not in st.session_state:
    st.session_state.reading_history = []
if 'current_user' not in st.session_state:
    st.session_state.current_user = f"user_{random.randint(1000, 9999)}"
if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []

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

    try:
        response = requests.post(
            f"{API_URL}/recommend", json=payload, timeout=30)
        if response.status_code == 200:
            # Log this recommendation session
            st.session_state.recommendation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "query": query,
                "num_results": k,
                "articles": response.json()
            })
            return response.json()
    except Exception as e:
        st.error(f"Error: {e}")
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


def log_interaction(article_id, action, user_id=None):
    """Log user interaction locally"""
    if user_id is None:
        user_id = st.session_state.current_user

    interaction = {
        "user_id": user_id,
        "article_id": article_id,
        "action": action,
        "timestamp": datetime.now().isoformat()
    }

    st.session_state.user_interactions.append(interaction)

    # Update ratings
    if action in ["like", "dislike"]:
        rating = 1 if action == "like" else -1
        st.session_state.article_ratings[article_id] = rating

    # Also send to API if available
    try:
        requests.post(f"{API_URL}/log_interaction",
                      json=interaction, timeout=2)
    except:
        pass  # API might not have this endpoint yet

# Display functions


def display_article_interactive(article, index):
    """Display article with interactive buttons"""

    # Get user rating for this article
    user_rating = st.session_state.article_ratings.get(article['news_id'], 0)

    with st.container():
        st.markdown(f"""
        <div class="article-card">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div style="flex: 1;">
                    <h3 style="margin-top: 0;">{index}. {article['title']}</h3>
                    <div>
                        <span class="category-badge">{article['category'].upper()}</span>
                        {f"<span class='category-badge' style='background: #e8f5e8; color: #2e7d32;'>{article.get('subcategory', '').upper()}</span>" if article.get('subcategory') else ""}
                    </div>
                    <p style="color: #555; margin: 1rem 0;">{article.get('abstract', 'No abstract available')[:200]}...</p>
                    <div style="display: flex; gap: 1rem; color: #666; font-size: 0.9rem;">
                        <span>👁️ {article.get('impressions', 0):,} views</span>
                        <span>📊 {article.get('ctr', 0):.1%} CTR</span>
                        <span>🎯 Score: {article.get('score', 0):.2f}</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Interactive buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            like_disabled = user_rating == 1
            like_label = "👍 Liked" if like_disabled else "👍 Like"
            if st.button(like_label, key=f"like_{article['news_id']}_{index}",
                         disabled=like_disabled,
                         use_container_width=True):
                log_interaction(article['news_id'], "like")
                st.success("Thanks for your feedback!")
                time.sleep(0.5)
                st.rerun()

        with col2:
            dislike_disabled = user_rating == -1
            dislike_label = "👎 Disliked" if dislike_disabled else "👎 Dislike"
            if st.button(dislike_label, key=f"dislike_{article['news_id']}_{index}",
                         disabled=dislike_disabled,
                         use_container_width=True):
                log_interaction(article['news_id'], "dislike")
                st.warning("Noted! We'll show fewer similar articles.")
                time.sleep(0.5)
                st.rerun()

        with col3:
            if st.button("📖 Read Article", key=f"read_{article['news_id']}_{index}",
                         use_container_width=True):
                log_interaction(article['news_id'], "read")
                st.session_state.reading_history.append({
                    "article_id": article['news_id'],
                    "title": article['title'],
                    "timestamp": datetime.now().isoformat()
                })

                # Show expanded article view
                with st.expander("📄 Full Article View", expanded=True):
                    st.markdown(f"### {article['title']}")
                    st.markdown(f"**Category:** {article['category']}")
                    if article.get('subcategory'):
                        st.markdown(
                            f"**Subcategory:** {article['subcategory']}")
                    st.markdown("---")
                    st.markdown(
                        f"{article.get('abstract', 'No content available')}")
                    st.markdown("---")
                    st.markdown(f"*Stats:* {article.get('impressions', 0):,} views | "
                                f"{article.get('ctr', 0):.1%} CTR | "
                                f"Recommendation score: {article.get('score', 0):.2f}")

        st.markdown("---")


def display_user_analytics():
    """Display user interaction analytics"""
    if not st.session_state.user_interactions:
        return

    st.sidebar.markdown("### 📊 Your Activity")

    # Count interactions
    likes = sum(
        1 for i in st.session_state.user_interactions if i['action'] == 'like')
    dislikes = sum(
        1 for i in st.session_state.user_interactions if i['action'] == 'dislike')
    reads = sum(
        1 for i in st.session_state.user_interactions if i['action'] == 'read')

    st.sidebar.metric("👍 Likes", likes)
    st.sidebar.metric("👎 Dislikes", dislikes)
    st.sidebar.metric("📖 Reads", reads)

    # Show recent activity
    if st.sidebar.checkbox("Show Recent Activity"):
        recent = st.session_state.user_interactions[-5:]
        for act in reversed(recent):
            action_icon = {
                'like': '👍',
                'dislike': '👎',
                'read': '📖'
            }.get(act['action'], '📝')

            time_str = datetime.fromisoformat(
                act['timestamp']).strftime("%H:%M")
            st.sidebar.text(
                f"{action_icon} {act['action'].title()} at {time_str}")


def display_system_feedback():
    """Show how system learns from feedback"""
    if st.session_state.user_interactions:
        st.sidebar.markdown("### 🤖 System Learning")
        st.sidebar.info("""
        The ML model is learning from your feedback:
        - 👍 Likes: Reinforce similar content
        - 👎 Dislikes: Reduce similar content
        - 📖 Reads: Understand your reading patterns
        """)

        if st.sidebar.button("Reset My Preferences"):
            st.session_state.user_interactions = []
            st.session_state.article_ratings = {}
            st.session_state.reading_history = []
            st.sidebar.success("Preferences reset!")
            st.rerun()

# Main app


def main():
    st.markdown('<p class="main-header">📰 Interactive News Recommender</p>',
                unsafe_allow_html=True)
    st.markdown("Your personalized news feed that learns from your interactions")

    # Check API
    api_healthy = check_api_health()

    if not api_healthy:
        st.warning("⚠️ API is not running. Some features may be limited.")
        st.info("Start the API with: `python app/api.py`")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # User info
        st.markdown(f"**Current User:** `{st.session_state.current_user}`")

        # Mode selection
        mode = st.radio(
            "Recommendation Mode",
            ["🔍 Search Mode", "👤 Personalized Mode",
                "🔥 Trending Mode", "🎲 Discovery Mode"],
            help="Different strategies for finding articles"
        )

        st.markdown("---")

        if mode == "🔍 Search Mode":
            query = st.text_input(
                "Search Query",
                placeholder="e.g., AI technology, sports news, politics...",
                help="Find articles by keywords or topics"
            )
            user_id = None

        elif mode == "👤 Personalized Mode":
            sample_users = get_sample_users() if api_healthy else []

            if sample_users:
                user_id = st.selectbox(
                    "Select User",
                    [""] + sample_users,
                    help="Choose a user profile for personalized recommendations"
                )
            else:
                user_id = st.text_input(
                    "Enter User ID:", value=st.session_state.current_user)

            query = None

        elif mode == "🔥 Trending Mode":
            st.info("Shows currently popular articles based on user engagement")
            user_id = st.session_state.current_user
            query = "trending"

        else:  # Discovery Mode
            st.info("Explores diverse articles outside your usual interests")
            user_id = st.session_state.current_user
            query = "discovery"

        st.markdown("---")

        # Number of results
        num_results = st.slider(
            "Number of Articles",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )

        # Refresh rate
        auto_refresh = st.checkbox("Auto-refresh every 30s", value=False)

        st.markdown("---")

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
                st.session_state.get_recs = True
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

    # Display user analytics
    display_user_analytics()
    display_system_feedback()

    # Main content area
    if 'get_recs' in st.session_state and st.session_state.get_recs:
        if (mode == "🔍 Search Mode" and not query) or (mode == "👤 Personalized Mode" and not user_id):
            st.warning("Please provide a search query or select a user")
        else:
            with st.spinner("🤖 Finding the best articles for you..."):
                recommendations = get_recommendations(
                    user_id=user_id if mode in [
                        "👤 Personalized Mode", "🔥 Trending Mode", "🎲 Discovery Mode"] else None,
                    query=query if mode == "🔍 Search Mode" else None,
                    k=num_results
                )

            if recommendations:
                st.success(f"✅ Found {len(recommendations)} recommendations")

                # Show mode info
                mode_info = {
                    "🔍 Search Mode": f"Search results for: **'{query}'**",
                    "👤 Personalized Mode": f"Personalized for user: **'{user_id}'**",
                    "🔥 Trending Mode": "🔥 **Currently Trending Articles**",
                    "🎲 Discovery Mode": "🎲 **Explore New Topics**"
                }
                st.markdown(f"### {mode_info[mode]}")

                # Display articles
                for i, article in enumerate(recommendations, 1):
                    display_article_interactive(article, i)

                # Show feedback summary
                if st.session_state.user_interactions:
                    with st.expander("📈 Your Feedback Summary"):
                        df_interactions = pd.DataFrame(
                            st.session_state.user_interactions)
                        if not df_interactions.empty:
                            st.dataframe(
                                df_interactions[['timestamp', 'action', 'article_id']].tail(10))

                # Auto-refresh logic
                if auto_refresh:
                    time.sleep(30)
                    st.rerun()
            else:
                st.error(
                    "No recommendations found. Try different search terms or user.")
    else:
        # Welcome screen
        st.markdown("""
        ## 🎯 Welcome to the Interactive News Recommender!
        
        This system learns from your interactions to provide better recommendations over time.
        
        ### How it works:
        1. **Choose a mode** in the sidebar
        2. **Get recommendations** based on your selection
        3. **Interact with articles** using Like/Dislike/Read buttons
        4. **Watch the system learn** from your feedback
        
        ### Features:
        - **Personalized Recommendations**: Based on user history
        - **Search**: Find articles by keywords
        - **Trending**: See what's popular right now
        - **Discovery**: Explore new topics
        - **Interactive Learning**: System improves with your feedback
        
        ### 🚀 Ready to start?
        Configure your settings in the sidebar and click **"Get Recommendations"**!
        """)

        # Quick demo
        if st.button("🎮 Try Quick Demo"):
            with st.spinner("Loading demo recommendations..."):
                demo_recs = [
                    {
                        "news_id": "demo_1",
                        "title": "AI Breakthrough in Healthcare: New Model Detects Diseases with 99% Accuracy",
                        "category": "technology",
                        "subcategory": "health",
                        "abstract": "Researchers have developed an AI system that can detect early signs of diseases from medical images with unprecedented accuracy, potentially revolutionizing diagnostic medicine.",
                        "score": 0.95,
                        "ctr": 0.25,
                        "impressions": 1500
                    },
                    {
                        "news_id": "demo_2",
                        "title": "Renewable Energy Investments Hit Record High in 2024",
                        "category": "finance",
                        "subcategory": "energy",
                        "abstract": "Global investments in solar and wind power have surged by 45% this year, signaling a major shift towards sustainable energy sources.",
                        "score": 0.88,
                        "ctr": 0.18,
                        "impressions": 1200
                    }
                ]

                st.success("Demo loaded! Try interacting with these articles:")
                for i, article in enumerate(demo_recs, 1):
                    display_article_interactive(article, i)


if __name__ == "__main__":
    main()
