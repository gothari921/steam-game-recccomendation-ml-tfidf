import streamlit as st
import joblib
import pandas as pd
import numpy as np
import difflib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Steam Game Recommender", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 20px;
    }
    .game-card {
        background-color: #1f1f1f;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #00a8e8;
    }
    .header {
        color: #00a8e8;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all joblib model components"""
    vectorizer = joblib.load("model_vectorizer.joblib")
    feature_vectors = joblib.load("model_vectors.joblib")
    indices = joblib.load("model_indices.joblib")
    df = joblib.load("model_df.joblib")
    return vectorizer, feature_vectors, indices, df

def recommend_games(game_name, vectorizer, feature_vectors, indices, df, top_n=10):
    """Get game recommendations based on similarity"""
    close_match = difflib.get_close_matches(game_name, df['name'], n=1, cutoff=0.3)
    
    if not close_match:
        return None, f"Game '{game_name}' not found in database."
    
    matched_name = close_match[0]
    
    if matched_name not in indices.index:
        return None, f"Could not find exact match for '{game_name}'."
    
    idx = indices[matched_name]
    
    # Calculate similarity scores
    sim_scores = cosine_similarity(
        feature_vectors[idx],
        feature_vectors
    ).flatten()
    
    # Get top similar games (excluding the game itself)
    similar_indices = sim_scores.argsort()[::-1][1:top_n+1]
    
    recommended_games = df.loc[similar_indices, ['name', 'popular_tags', 'developer', 'publisher']].copy()
    recommended_games['similarity'] = sim_scores[similar_indices]
    
    return matched_name, recommended_games

# Main UI
st.markdown("<h1 class='header'>🎮 Steam Game Recommender</h1>", unsafe_allow_html=True)
st.markdown("Find games similar to your favorites using TF-IDF and Cosine Similarity")

# Load models
try:
    vectorizer, feature_vectors, indices, df = load_models()
    total_games = len(df)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Sidebar configuration
st.sidebar.markdown("## Settings")
top_n = st.sidebar.slider("Number of recommendations", 5, 20, 10)
similarity_threshold = st.sidebar.slider("Show only games with similarity > %", 0.0, 1.0, 0.1)

st.sidebar.markdown(f"### Database Info")
st.sidebar.info(f"Total games in database: **{total_games}**")

# Main input area
col1, col2 = st.columns([3, 1])

with col1:
    game_input = st.text_input(
        "Enter a game name:",
        placeholder="e.g., 'Elden Ring', 'Dark Souls', 'Ghost of Tsushima'",
        label_visibility="collapsed"
    )

with col2:
    search_button = st.button("Search", use_container_width=True)

# Display results
if game_input or search_button:
    if game_input:
        matched_game, result = recommend_games(
            game_input, 
            vectorizer, 
            feature_vectors, 
            indices, 
            df, 
            top_n=top_n
        )
        
        if matched_game is None:
            st.warning(result)
        else:
            st.success(f"Found match: **{matched_game}**")
            
            # Filter by similarity threshold
            if isinstance(result, pd.DataFrame):
                filtered_result = result[result['similarity'] > similarity_threshold]
                
                if len(filtered_result) == 0:
                    st.info("No games found with the selected similarity threshold. Try lowering it.")
                else:
                    st.markdown("---")
                    st.markdown(f"### Top {len(filtered_result)} Recommended Games")
                    
                    for idx, (_, game) in enumerate(filtered_result.iterrows(), 1):
                        similarity_pct = int(game['similarity'] * 100)
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"""
<div class='game-card'>
    <strong>{idx}. {game['name']}</strong><br>
    <small>Developer:</small> {game['developer']}<br>
    <small>Publisher:</small> {game['publisher']}<br>
    <small>Tags:</small> {game['popular_tags'][:50]}...
</div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("Similarity", f"{similarity_pct}%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
    Built with Streamlit | TF-IDF + Cosine Similarity | Steam Game Dataset
</div>
""", unsafe_allow_html=True)
