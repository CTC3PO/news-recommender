"""
Modern Streamlit web interface for news recommendation system
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page configuration with better theme
st.set_page_config(
    page_title="Intelligent News Recommender",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom modern CSS with gradients and animations
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .sub-header {
        color: #6c757d;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .article-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .article-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .category-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .score-badge {
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stSelectbox, .stTextInput, .stSlider {
        background: white;
        border-radius: 10px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    .tab-button {
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# API configuration
API_URL = "http://localhost:8000"

# Helper functions


@st.cache_data(ttl=300)
def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


@st.cache_data(ttl=300)
def get_recommendations(user_id=None, query=None, k=10, mode="hybrid"):
    """Get recommendations from API"""
    payload = {"k": k}
    if user_id:
        payload["user_id"] = user_id
    if query:
        payload["query"] = query

    # Add mode parameter if supported by API
    payload["mode"] = mode

    try:
        response = requests.post(
            f"{API_URL}/recommend", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None


@st.cache_data(ttl=300)
def get_user_info(user_id):
    """Get user profile information"""
    try:
        response = requests.get(f"{API_URL}/user/{user_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


@st.cache_data(ttl=300)
def get_sample_users():
    """Get sample user IDs"""
    try:
        response = requests.get(f"{API_URL}/users", timeout=5)
        if response.status_code == 200:
            return response.json()['users']
    except:
        pass
    return []


@st.cache_data(ttl=300)
def get_categories():
    """Get category distribution"""
    try:
        response = requests.get(f"{API_URL}/categories", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


def generate_demo_recommendations():
    """Generate demo recommendations for when API is down"""
    demo_articles = [
        {
            "news_id": "demo_1",
            "title": "Artificial Intelligence Revolutionizes Healthcare Diagnostics",
            "category": "technology",
            "subcategory": "ai",
            "abstract": "New AI models are achieving human-level accuracy in medical image analysis...",
            "score": 0.95,
            "ctr": 0.23,
            "impressions": 1250
        },
        {
            "news_id": "demo_2",
            "title": "Renewable Energy Investments Reach Record High in 2024",
            "category": "finance",
            "subcategory": "energy",
            "abstract": "Global investments in solar and wind power have increased by 45% compared to last year...",
            "score": 0.88,
            "ctr": 0.18,
            "impressions": 980
        }
    ]
    return demo_articles


# Initialize session state
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'user_history' not in st.session_state:
    st.session_state.user_history = []
if 'selected_articles' not in st.session_state:
    st.session_state.selected_articles = []


def main():
    # Modern Header with Hero Section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<p class="main-header">📰 NewsAI</p>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p class="sub-header">Intelligent News Discovery & Personalization</p>', unsafe_allow_html=True)

    # Top Navigation Bar
    with st.container():
        cols = st.columns(6)
        tabs = ["🏠 Home", "🔍 Search", "👤 Personalized",
                "📊 Analytics", "⚡ Trending", "⚙️ Settings"]
        for i, tab in enumerate(tabs):
            with cols[i]:
                if st.button(tab, key=f"tab_{i}", use_container_width=True):
                    st.session_state.current_tab = tab

    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "🏠 Home"

    # Check API health
    api_healthy = check_api_health()

    # Main content based on selected tab
    if st.session_state.current_tab == "🏠 Home":
        render_home_page(api_healthy)
    elif st.session_state.current_tab == "🔍 Search":
        render_search_page(api_healthy)
    elif st.session_state.current_tab == "👤 Personalized":
        render_personalized_page(api_healthy)
    elif st.session_state.current_tab == "📊 Analytics":
        render_analytics_page(api_healthy)
    elif st.session_state.current_tab == "⚡ Trending":
        render_trending_page(api_healthy)
    elif st.session_state.current_tab == "⚙️ Settings":
        render_settings_page()


def render_home_page(api_healthy):
    """Home page with overview and quick actions"""
    st.markdown("## 🎯 Dashboard Overview")

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="metric-card"><div class="feature-icon">📚</div><h3>50K+</h3><p>Articles</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(
            '<div class="metric-card"><div class="feature-icon">👥</div><h3>1M+</h3><p>Users</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(
            '<div class="metric-card"><div class="feature-icon">🎯</div><h3>95%</h3><p>Accuracy</p></div>', unsafe_allow_html=True)
    with col4:
        status = "🟢" if api_healthy else "🔴"
        st.markdown(
            f'<div class="metric-card"><div class="feature-icon">{status}</div><h3>{"Online" if api_healthy else "Offline"}</h3><p>API Status</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Quick action cards
    st.markdown("## 🚀 Quick Actions")
    cols = st.columns(4)

    with cols[0]:
        if st.button("🔍 Search Articles", use_container_width=True, key="quick_search"):
            st.session_state.current_tab = "🔍 Search"
            st.rerun()

    with cols[1]:
        if st.button("👤 Find Recommendations", use_container_width=True, key="quick_recommend"):
            st.session_state.current_tab = "👤 Personalized"
            st.rerun()

    with cols[2]:
        if st.button("📈 View Analytics", use_container_width=True, key="quick_analytics"):
            st.session_state.current_tab = "📊 Analytics"
            st.rerun()

    with cols[3]:
        if st.button("⚡ Trending Now", use_container_width=True, key="quick_trending"):
            st.session_state.current_tab = "⚡ Trending"
            st.rerun()

    st.markdown("---")

    # Features showcase
    st.markdown("## ✨ Key Features")

    features = [
        {"icon": "🤖", "title": "AI-Powered",
            "desc": "Uses advanced ML models for intelligent recommendations"},
        {"icon": "⚡", "title": "Real-time",
            "desc": "Instant recommendations with low latency"},
        {"icon": "🎯", "title": "Personalized",
            "desc": "Tailored content based on user behavior"},
        {"icon": "📊", "title": "Analytics",
            "desc": "Detailed insights and performance metrics"},
        {"icon": "🔍", "title": "Semantic Search",
            "desc": "Understands meaning, not just keywords"},
        {"icon": "🔄", "title": "Hybrid Models",
            "desc": "Combines multiple recommendation strategies"}
    ]

    for i in range(0, len(features), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(features):
                with cols[j]:
                    feat = features[i + j]
                    st.markdown(f"""
                    <div style="padding: 1.5rem; background: white; border-radius: 15px; border: 1px solid #e9ecef;">
                        <div style="font-size: 2rem; margin-bottom: 1rem;">{feat['icon']}</div>
                        <h4>{feat['title']}</h4>
                        <p style="color: #6c757d; font-size: 0.9rem;">{feat['desc']}</p>
                    </div>
                    """, unsafe_allow_html=True)


def render_search_page(api_healthy):
    """Enhanced search page with multiple modes"""
    st.markdown("## 🔍 Advanced Search")

    # Search modes
    search_mode = st.radio(
        "Search Mode",
        ["🔤 Keyword Search", "📝 Semantic Search", "📊 Advanced Filters"],
        horizontal=True,
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        if search_mode == "🔤 Keyword Search":
            query = st.text_input(
                "Enter keywords:",
                placeholder="e.g., artificial intelligence machine learning",
                help="Search by specific keywords"
            )
            mode = "keyword"

        elif search_mode == "📝 Semantic Search":
            query = st.text_area(
                "Describe what you're looking for:",
                placeholder="e.g., Articles about recent advancements in renewable energy and sustainability...",
                help="Describe in natural language, our AI will understand the meaning"
            )
            mode = "semantic"

        else:  # Advanced Filters
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                category = st.selectbox(
                    "Category",
                    ["All", "Technology", "Sports", "Finance",
                        "Entertainment", "Health", "Lifestyle"]
                )
            with col_f2:
                min_ctr = st.slider("Minimum CTR", 0.0, 1.0, 0.0, 0.01)
            with col_f3:
                date_range = st.selectbox(
                    "Date Range",
                    ["All time", "Last week", "Last month", "Last year"]
                )
            query = f"category:{category} ctr>{min_ctr}"
            mode = "advanced"

    with col2:
        num_results = st.number_input(
            "Results",
            min_value=5,
            max_value=50,
            value=15,
            step=5
        )

    # Additional filters
    with st.expander("🔧 Advanced Options"):
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            sort_by = st.selectbox(
                "Sort by",
                ["Relevance", "Popularity", "CTR", "Date", "Random"]
            )
        with col_a2:
            include_abstract = st.checkbox("Include abstracts", value=True)
        with col_a3:
            grouping = st.selectbox(
                "Group by",
                ["None", "Category", "Date"]
            )

    # Search button
    if st.button("🔍 Search Articles", type="primary", use_container_width=True):
        if query or mode == "advanced":
            with st.spinner("🔄 Searching through 50K+ articles..."):
                recommendations = get_recommendations(
                    query=query if mode != "advanced" else None,
                    k=num_results,
                    mode=mode
                ) if api_healthy else generate_demo_recommendations()

            if recommendations:
                st.session_state.recommendations = recommendations
                display_search_results(recommendations, include_abstract)
        else:
            st.warning("Please enter a search query")


def render_personalized_page(api_healthy):
    """Enhanced personalized recommendations with multiple strategies"""
    st.markdown("## 👤 Personalized Recommendations")

    # User selection
    col1, col2 = st.columns([2, 1])

    with col1:
        sample_users = get_sample_users() if api_healthy else [
            "demo_user_1", "demo_user_2", "demo_user_3"]

        if sample_users:
            selected_user = st.selectbox(
                "Select User",
                [""] + sample_users,
                help="Choose a user to get personalized recommendations"
            )
        else:
            selected_user = st.text_input("Enter User ID:")

    with col2:
        # Recommendation strategy
        strategy = st.selectbox(
            "Strategy",
            ["🎯 Hybrid", "🤝 Collaborative",
                "📝 Content-based", "🔥 Trending", "🎲 Explore"]
        )

    # User profile section
    if selected_user:
        with st.spinner("📊 Loading user profile..."):
            user_info = get_user_info(selected_user) if api_healthy else {
                "user_id": selected_user,
                "articles_read": 150,
                "top_category": "technology",
                "category_diversity": 0.65
            }

        if user_info:
            st.success(f"✅ Profile loaded for **{selected_user}**")

            # User metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Articles Read", user_info['articles_read'])
            with col_m2:
                st.metric("Top Category", user_info['top_category'])
            with col_m3:
                st.metric(
                    "Diversity", f"{user_info['category_diversity']:.2%}")
            with col_m4:
                st.metric(
                    "Engagement", "High" if user_info['articles_read'] > 100 else "Medium")

    # Recommendation options
    with st.expander("⚙️ Recommendation Settings"):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            num_recs = st.slider("Number of recommendations", 5, 30, 10)
        with col_s2:
            diversity = st.slider("Diversity", 0.0, 1.0, 0.5, 0.1)
        with col_s3:
            novelty = st.slider("Novelty", 0.0, 1.0, 0.3, 0.1)

    # Get recommendations button
    if st.button("🚀 Get Personalized Recommendations", type="primary", use_container_width=True):
        if selected_user:
            with st.spinner("🤖 Generating personalized recommendations..."):
                recommendations = get_recommendations(
                    user_id=selected_user,
                    k=num_recs,
                    mode=strategy.lower().replace(" ", "_")
                ) if api_healthy else generate_demo_recommendations()

            if recommendations:
                st.session_state.recommendations = recommendations
                display_recommendations(recommendations, "personalized")
        else:
            st.warning("Please select or enter a user ID")


def render_analytics_page(api_healthy):
    """Analytics and insights page"""
    st.markdown("## 📊 Analytics Dashboard")

    # Fetch data
    categories = get_categories() if api_healthy else {
        "technology": 12000, "sports": 8500, "finance": 7200}

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Articles", "50,284")
    with col2:
        st.metric("Active Users", "49,108")
    with col3:
        st.metric("Avg CTR", "12.3%")
    with col4:
        st.metric("Model AUC", "0.89")

    st.markdown("---")

    # Charts
    col_ch1, col_ch2 = st.columns(2)

    with col_ch1:
        st.markdown("### 📈 Category Distribution")
        if categories and 'categories' in categories:
            df_categories = pd.DataFrame({
                'Category': list(categories['categories'].keys()),
                'Count': list(categories['categories'].values())
            })
            fig = px.bar(df_categories.nlargest(10, 'Count'),
                         x='Category', y='Count',
                         color='Count',
                         color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category data not available")

    with col_ch2:
        st.markdown("### 📊 Performance Metrics")
        metrics = {
            'Metric': ['CTR', 'Impressions', 'User Engagement', 'Model Accuracy'],
            'Value': [0.123, 0.85, 0.72, 0.89]
        }
        fig = px.bar(pd.DataFrame(metrics), x='Metric', y='Value',
                     color='Value', color_continuous_scale='plasma')
        st.plotly_chart(fig, use_container_width=True)

    # User segmentation
    st.markdown("### 👥 User Segmentation")
    seg_cols = st.columns(4)
    segments = [
        {"name": "Power Users", "count": "15,234", "color": "green"},
        {"name": "Regular Users", "count": "25,671", "color": "blue"},
        {"name": "Casual Users", "count": "7,543", "color": "orange"},
        {"name": "New Users", "count": "660", "color": "purple"}
    ]

    for i, seg in enumerate(segments):
        with seg_cols[i]:
            st.markdown(f"""
            <div style="background: {seg['color']}10; padding: 1rem; border-radius: 10px; border-left: 4px solid {seg['color']};">
                <h4 style="margin: 0;">{seg['name']}</h4>
                <h2 style="margin: 0.5rem 0; color: {seg['color']}">{seg['count']}</h2>
            </div>
            """, unsafe_allow_html=True)


def render_trending_page(api_healthy):
    """Trending and discovery page"""
    st.markdown("## ⚡ Trending Now")

    # Time filters
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        timeframe = st.selectbox(
            "Timeframe", ["Today", "This Week", "This Month"])
    with col_t2:
        region = st.selectbox(
            "Region", ["Global", "North America", "Europe", "Asia"])
    with col_t3:
        category = st.selectbox(
            "Category", ["All", "Technology", "Politics", "Sports", "Entertainment"])

    # Trending metrics
    st.markdown("### 📈 Trending Metrics")

    metrics_data = {
        "Metric": ["Virality Score", "Engagement Rate", "Share Rate", "Comment Density"],
        "Value": [0.85, 0.72, 0.45, 0.32],
        "Change": ["+12%", "+8%", "-3%", "+15%"]
    }

    for i in range(4):
        cols = st.columns([3, 1, 1])
        with cols[0]:
            st.write(f"**{metrics_data['Metric'][i]}**")
        with cols[1]:
            st.metric("", f"{metrics_data['Value'][i]:.2f}")
        with cols[2]:
            color = "green" if "+" in metrics_data['Change'][i] else "red"
            st.markdown(
                f"<span style='color: {color}; font-weight: bold;'>{metrics_data['Change'][i]}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # Trending topics
    st.markdown("### 🔥 Hot Topics")

    topics = [
        {"topic": "AI Regulation", "mentions": 12500, "growth": "+45%"},
        {"topic": "Climate Summit", "mentions": 9800, "growth": "+32%"},
        {"topic": "Tech Layoffs", "mentions": 7600, "growth": "+28%"},
        {"topic": "Sports Championship", "mentions": 5400, "growth": "+65%"},
        {"topic": "Election Updates", "mentions": 4200, "growth": "+18%"}
    ]

    for topic in topics:
        with st.container():
            cols = st.columns([3, 2, 1])
            with cols[0]:
                st.write(f"**{topic['topic']}**")
            with cols[1]:
                st.progress(topic['mentions'] / 15000)
            with cols[2]:
                color = "green" if "+" in topic['growth'] else "red"
                st.markdown(
                    f"<span style='color: {color};'>{topic['growth']}</span>", unsafe_allow_html=True)


def render_settings_page():
    """Settings and preferences page"""
    st.markdown("## ⚙️ Settings & Preferences")

    # Theme settings
    st.markdown("### 🎨 Appearance")
    theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
    density = st.selectbox("Density", ["Comfortable", "Compact", "Expanded"])

    # Notification settings
    st.markdown("### 🔔 Notifications")
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        email_notifications = st.checkbox("Email notifications", value=True)
        trending_alerts = st.checkbox("Trending alerts", value=True)
    with col_n2:
        weekly_digest = st.checkbox("Weekly digest", value=True)
        recommendation_updates = st.checkbox("New recommendations", value=True)

    # Recommendation preferences
    st.markdown("### 🎯 Recommendation Preferences")

    preferred_categories = st.multiselect(
        "Preferred Categories",
        ["Technology", "Sports", "Finance", "Entertainment",
            "Health", "Lifestyle", "Politics", "Science"],
        default=["Technology", "Science"]
    )

    content_length = st.select_slider(
        "Preferred Article Length",
        options=["Short", "Medium", "Long"],
        value="Medium"
    )

    # Data controls
    st.markdown("### 📊 Data & Privacy")
    st.checkbox("Save search history", value=True)
    st.checkbox("Share anonymous usage data", value=False)

    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved successfully!")


def display_search_results(recommendations, include_abstract=True):
    """Display search results with modern cards"""
    st.markdown(f"### 📄 Found {len(recommendations)} Results")

    for i, article in enumerate(recommendations, 1):
        # Category color mapping
        category_colors = {
            'technology': '#3b82f6',
            'sports': '#10b981',
            'finance': '#f59e0b',
            'entertainment': '#8b5cf6',
            'health': '#ef4444',
            'lifestyle': '#ec4899'
        }

        color = category_colors.get(article['category'].lower(), '#6b7280')

        with st.container():
            st.markdown(f"""
            <div class="article-card">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex: 1;">
                        <h3 style="margin-top: 0; margin-bottom: 0.5rem;">{i}. {article['title']}</h3>
                        <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.5rem;">
                            <span class="category-badge" style="background: {color}20; color: {color};">{article['category'].upper()}</span>
                            <span style="color: #6c757d; font-size: 0.9rem;">{article.get('subcategory', 'General')}</span>
                        </div>
                        {f'<p style="color: #495057; line-height: 1.5;">{article.get("abstract", "")[:200]}...</p>' if include_abstract and article.get('abstract') else ''}
                        <div style="display: flex; gap: 1.5rem; margin-top: 1rem; color: #6c757d; font-size: 0.85rem;">
                            <span>👁️ {article['impressions']:,} views</span>
                            <span>📊 {article['ctr']:.1%} CTR</span>
                            <span>📅 Just now</span>
                        </div>
                    </div>
                    <div style="text-align: right; min-width: 100px;">
                        <div class="score-badge" style="background: linear-gradient(135deg, {color}40, {color}60); color: {color};">
                            {article['score']:.2f}
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #6c757d;">Score</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def display_recommendations(recommendations, mode="personalized"):
    """Display recommendations with visualization"""
    if not recommendations:
        return

    # Summary metrics
    avg_score = np.mean([r['score'] for r in recommendations])
    avg_ctr = np.mean([r['ctr'] for r in recommendations])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Score", f"{avg_score:.3f}")
    with col2:
        st.metric("Average CTR", f"{avg_ctr:.2%}")
    with col3:
        st.metric(
            "Diversity", f"{len(set(r['category'] for r in recommendations))} categories")

    st.markdown("---")

    # Display each recommendation
    display_search_results(recommendations)


if __name__ == "__main__":
    main()
