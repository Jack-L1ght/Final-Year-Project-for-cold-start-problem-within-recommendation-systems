
import os
import pandas as pd
import joblib
from sklearn.preprocessing import normalize

print("Loading recommender models...")


svd_model = joblib.load("models/svd_baseline.pkl")


tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
movie_vectors = joblib.load("models/movie_vectors.pkl")
movie_index = joblib.load("models/movie_index.pkl")

user_profiles_path = "models/user_profiles.pkl"

if not os.path.exists(user_profiles_path):
    print("User profiles file not found → creating new one")
    user_profiles = {}
else:
    try:
        user_profiles = joblib.load(user_profiles_path)
        print("User profiles loaded:", len(user_profiles))
    except:
        print("User profiles corrupted → resetting file")
        user_profiles = {}
        joblib.dump(user_profiles, user_profiles_path)


movies = pd.read_csv("Filtered_Data/movies.csv")

movie_title_lookup = dict(zip(movies.movieId, movies.title))

# Reduce for speed
candidate_movies = movies.sample(1000)["movieId"].values

print("Models loaded successfully")

def create_user_from_genres(user_id, genres):

    global user_profiles

    print("Creating user:", user_id)
    print("Genres selected:", genres)

    genre_text = " ".join(genres)

    genre_vec = tfidf_vectorizer.transform([genre_text])

    genre_vec = normalize(genre_vec)

    user_profiles[user_id] = genre_vec

    joblib.dump(user_profiles, "models/user_profiles.pkl")

    print("User profile saved")

def add_user_rating(user_id, movie_id, rating):

    global user_profiles

    if movie_id not in movie_index:
        return

    movie_vec = movie_vectors[movie_index[movie_id]]

    if user_id not in user_profiles:
        user_profiles[user_id] = movie_vec * rating
    else:
        user_profiles[user_id] += movie_vec * rating

    joblib.dump(user_profiles, "models/user_profiles.pkl")

    print("User profile updated")

def predict_content(user_id, movie_id):

    if user_id not in user_profiles:
        return None

    if movie_id not in movie_index:
        return None

    user_vec = user_profiles[user_id]
    movie_vec = movie_vectors[movie_index[movie_id]]

    similarity = user_vec.dot(movie_vec.T)[0, 0]

    rating = 0.5 + similarity * (5.0 - 0.5)

    return rating

def predict_hybrid(user_id, movie_id, alpha=0.5):

    content_pred = predict_content(user_id, movie_id)

    try:
        svd_pred = svd_model.predict(user_id, movie_id).est
    except:
        svd_pred = None

    if svd_pred is not None and content_pred is not None:
        return alpha * svd_pred + (1 - alpha) * content_pred

    if svd_pred is not None:
        return svd_pred

    if content_pred is not None:
        return content_pred

    return None

def recommend_movies(user_id, top_n=10):

    print("Generating recommendations for user:", user_id)

    predictions = []

    for movie_id in candidate_movies:

        pred = predict_hybrid(user_id, movie_id)

        if pred is None:
            continue

        title = movie_title_lookup[movie_id]

        try:
            year = int(title[-5:-1]) if "(" in title else 2000
        except:
            year = 2000

        year_weight = (year - 1950) / (2025 - 1950)

        pred = pred * (0.7 + 0.3 * year_weight)

        predictions.append((movie_id, pred))

    predictions.sort(key=lambda x: x[1], reverse=True)

    top_movies = predictions[:top_n]

    results = []

    for movie_id, score in top_movies:

        results.append({
            "movieId": movie_id,
            "title": movie_title_lookup[movie_id],
            "predicted_rating": round(score, 2)
        })

    return results
