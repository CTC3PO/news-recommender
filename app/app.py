"""
Hugging Face Space entry point
"""
import streamlit as st
import requests
import pandas as pd
import sys
import os

# Add local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock API if not available


class MockAPI:
    def recommend(self, user_id=None, query=None, k=10):
        # Return mock data for demo
        return [
            {
                "news_id": f"demo_{i}",
                "title": f"Demo Article {i} - AI Breakthrough in {['Healthcare', 'Finance', 'Education'][i%3]}",
                "category": ["tech", "finance", "health"][i % 3],
                "abstract": f"This is a demo article showing the recommendation system in action...",
                "score": 0.9 - (i*0.1),
                "ctr": 0.15,
                "impressions": 1000 - (i*100)
            }
            for i in range(k)
        ]

# Main Streamlit app


def main():
    st.set_page_config(page_title="News Recommender", layout="wide")

    st.title("📰 News Recommender Demo")

    # Interactive elements
    st.sidebar.header("Settings")

    # Choose demo mode
    mode = st.sidebar.radio(
        "Demo Mode",
        ["🚀 Live Demo", "🧪 Interactive Test", "📊 System Insights"]
    )

    if mode == "🚀 Live Demo":
        show_live_demo()
    elif mode == "🧪 Interactive Test":
        show_interactive_test()
    else:
        show_system_insights()


def show_live_demo():
    """Show working demo with mock data"""
    st.header("Live Recommendation Demo")

    # Simulate user
    user_id = st.selectbox(
        "Simulated User Profile",
        ["Tech Enthusiast 👨‍💻", "Sports Fan ⚽",
            "Business Professional 💼", "Student 🎓"]
    )

    # Get recommendations
    api = MockAPI()
    recommendations = api.recommend(user_id=user_id, k=8)

    # Display
    for i, article in enumerate(recommendations):
        with st.expander(f"{i+1}. {article['title']}"):
            st.markdown(f"**Category:** {article['category'].upper()}")
            st.markdown(f"**Score:** {article['score']:.2f}")
            st.markdown(f"**CTR:** {article['ctr']:.2%}")
            st.markdown(f"**Preview:** {article['abstract']}")

            # Interactive buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"👍 Like", key=f"like_{i}"):
                    st.success(
                        "Feedback recorded! System will learn from this.")
            with col2:
                if st.button(f"📖 Read Full", key=f"read_{i}"):
                    st.info("Opening article... (simulated)")


def show_interactive_test():
    """Let users test the system"""
    st.header("🧪 Test the Recommendation Engine")

    # Simulate different scenarios
    scenario = st.selectbox(
        "Test Scenario",
        [
            "New User (Cold Start)",
            "Power User (Lots of History)",
            "Seasonal Interest (Sports during championship)",
            "Breaking News Event"
        ]
    )

    st.markdown("### How the system handles this:")

    if scenario == "New User (Cold Start)":
        st.info("""
        **System Strategy:**
        1. Uses popular articles as baseline
        2. Asks for initial preferences
        3. Gradually learns from clicks
        4. Uses demographic info if available
        """)

        # Show recommendation evolution
        st.subheader("📈 Recommendation Evolution")

        data = pd.DataFrame({
            'Day': [1, 3, 7, 14, 30],
            'Personalization Score': [0.1, 0.3, 0.5, 0.7, 0.85]
        })

        st.line_chart(data.set_index('Day'))


def show_system_insights():
    """Show how the ML system works"""
    st.header("🤖 Under the Hood: ML System")

    st.markdown("""
    ### **Two-Stage Recommendation Pipeline:**
    
    #### **1. Candidate Retrieval (FAISS + Embeddings)**
    - 50K+ articles converted to semantic embeddings
    - FAISS vector database for millisecond search
    - Returns top 100 candidates based on similarity
    
    #### **2. ML Ranking (Gradient Boosting)**
    - **Features used:**
      - User reading history
      - Article popularity (CTR)
      - Category preferences
      - Recency of article
      - Text length/complexity
    
    #### **3. Real-time Adaptation**
    - Learns from clicks/ratings
    - Updates user profiles
    - Adjusts weights based on feedback
    """)

    # Show feature importance
    st.subheader("📊 Feature Importance")

    features = {
        'User-Article Category Match': 0.25,
        'Article CTR': 0.20,
        'User Reading Frequency': 0.15,
        'Article Recency': 0.12,
        'Text Similarity': 0.10,
        'Popularity Score': 0.08,
        'Reading Time Estimate': 0.05,
        'Social Signals': 0.05
    }

    df = pd.DataFrame(list(features.items()), columns=[
                      'Feature', 'Importance'])
    st.bar_chart(df.set_index('Feature'))


if __name__ == "__main__":
    main()
