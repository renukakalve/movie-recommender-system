import streamlit as st
import pickle
import requests
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- CONFIG --------------------
TMDB_API_KEY = "e6a7cfff7c5242b716f5ab4c5dd4dee5"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_movies():
    return pickle.load(open(os.path.join(BASE_DIR, "movies.pkl"), "rb"))

movies = load_movies()

# -------------------- COMPUTE SIMILARITY (ON THE FLY) --------------------
@st.cache_data
def compute_similarity(movies_df):
    cv = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )
    vectors = cv.fit_transform(movies_df["tags"]).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

similarity = compute_similarity(movies)

# -------------------- POSTER FETCH --------------------
def fetch_poster_by_title(title):
    url = (
        "https://api.themoviedb.org/3/search/movie"
        f"?api_key={TMDB_API_KEY}&query={title}"
    )

    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        if data["results"]:
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return POSTER_BASE_URL + poster_path

    except Exception:
        pass

    return None

# -------------------- RECOMMENDER --------------------
def recommend(movie_name):
    movie_index = movies[movies["title"] == movie_name].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_titles = []
    recommended_posters = []

    for idx, _ in movies_list:
        title = movies.iloc[idx]["title"]
        poster = fetch_poster_by_title(title)

        recommended_titles.append(title)
        recommended_posters.append(poster)

    return recommended_titles, recommended_posters

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].values
)

if st.button("Recommend"):
    titles, posters = recommend(selected_movie)

    st.subheader("Recommended Movies")

    cols = st.columns(5)   # 5 posters in one row

    for i in range(5):
        with cols[i]:
            if posters[i]:
                st.image(posters[i], use_container_width=True)
            else:
                st.write("No poster")
            st.caption(titles[i])
