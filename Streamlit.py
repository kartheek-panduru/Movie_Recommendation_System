import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config (must be first)
st.set_page_config(page_title="Movie Recommendation System", layout="centered")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('D:/python/Movie Recommendation System using ML/movies.csv')

movies_data = load_data()

# Preprocessing
def preprocess_data(df):
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        df[feature] = df[feature].fillna('')
    df['combined_features'] = df['genres'] + ' ' + df['keywords'] + ' ' + df['tagline'] + ' ' + df['cast'] + ' ' + df['director']
    return df

movies_data = preprocess_data(movies_data)

# Vectorization
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(movies_data['combined_features'])
similarity = cosine_similarity(feature_vectors)

# Recommendation Function
def recommend_movies(movie_name, movies_data, similarity):
    list_of_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_titles)
    
    if not find_close_match:
        return []
    
    close_match = find_close_match[0]
    index_of_movie = movies_data[movies_data.title == close_match].index[0]
    similarity_score = list(enumerate(similarity[index_of_movie]))
    sorted_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    recommended_movies = [movies_data.iloc[movie[0]]['title'] for movie in sorted_movies[1:21]]
    return recommended_movies

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Discover movies similar to your favorites!")

# Input and button
movie_name = st.text_input("Enter a movie name:", "")
get_recommendations = st.button("Get Recommendations")

# Trigger recommendation when pressing Enter or clicking the button
if movie_name and (get_recommendations or movie_name):  
    recommendations = recommend_movies(movie_name, movies_data, similarity)
    
    if recommendations:
        st.subheader("Movies recommended for you:")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")
    else:
        st.error("Movie not found. Please try another title.")

# Footer with friendly message
st.markdown("---")
st.markdown("🎬 Happy Watching! 🍿😃")
