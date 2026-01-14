import streamlit as st
import pickle
import requests
import urllib.parse
import time

# ================= CONFIG =================
TMDB_API_KEY = "e6a7cfff7c5242b716f5ab4c5dd4dee5"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500/"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}
# =========================================


@st.cache_data(show_spinner=False)
def fetch_poster_by_title(movie_title):
    """
    Safe poster fetch with timeout, retry, caching.
    Returns None if anything fails.
    """
    query = urllib.parse.quote(movie_title)

    url = (
        f"https://api.themoviedb.org/3/search/movie"
        f"?api_key={TMDB_API_KEY}&query={query}"
    )

    for _ in range(2):  # retry max 2 times
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=5
            )

            if response.status_code != 200:
                return None

            data = response.json()

            if not data.get("results"):
                return None

            poster_path = data["results"][0].get("poster_path")

            if poster_path is None:
                return None

            return POSTER_BASE_URL + poster_path

        except requests.exceptions.RequestException:
            time.sleep(1)  # wait before retry

    return None


# ============ LOAD DATA ===================
movies = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))
# ==========================================


def recommend(movie_name):
    movie_index = movies[movies["title"] == movie_name].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    titles = []
    posters = []

    for idx, _ in movies_list:
        title = movies.iloc[idx]["title"]
        titles.append(title)
        posters.append(fetch_poster_by_title(title))

    return titles, posters


# =============== STREAMLIT UI =================
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

