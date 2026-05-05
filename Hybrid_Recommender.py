# ---- Hybrid Recommender -----

# ---- Imports the required libraries ----

import numpy as np
import pandas as pd
import joblib

# ---- Load saved datasets -----

print("Loading filtered datasets...")

training_data = pd.read_csv("Filtered_Data/training_data.csv")
cold_users_test = pd.read_csv("Filtered_Data/cold_users_test.csv")
cold_movie_test = pd.read_csv("Filtered_Data/cold_movie_test.csv")

print("Datasets loaded")

# ---- Load trained models -----

print("Loading models...")

svd_model = joblib.load("models/svd_baseline.pkl")
movie_vectors = joblib.load("models/movie_vectors.pkl")
movie_index = joblib.load("models/movie_index.pkl")

try:
    user_profiles = joblib.load("models/user_profiles.pkl")
except:
    print("User profiles missing → using empty")
    user_profiles = {}

print("Models loaded")

# ---- Content-based prediction -----

def predict_content(user_id, movie_id):

    if user_id not in user_profiles:
        return None

    if movie_id not in movie_index:
        return None

    user_vec = user_profiles[user_id]
    movie_vec = movie_vectors[movie_index[movie_id]]

    sim = user_vec.dot(movie_vec.T)[0, 0]

    return 0.5 + sim * (5.0 - 0.5)

# ---- Hybrid prediction -----

def predict_hybrid(user_id, movie_id, alpha=0.5):

    content_pred = predict_content(user_id, movie_id)

    try:
        svd_pred = svd_model.predict(user_id, movie_id).est
    except:
        return None

    if content_pred is not None:
        return alpha * svd_pred + (1 - alpha) * content_pred

    return svd_pred

# ---- Evaluation of the model -----

print("\nEvaluating hybrid model...")

cold_users_sample = cold_users_test.sample(50000)
cold_movies_sample = cold_movie_test.sample(50000)

# ---- Cold users ----

predictions = []
truths = []

for i, row in enumerate(cold_users_sample.itertuples(index=False)):

    if i % 10000 == 0:
        print("Cold users processed:", i)

    prediction = predict_hybrid(row.userId, row.movieId)

    if prediction is not None:
        predictions.append(prediction)
        truths.append(row.rating)

if len(predictions) > 0:
    RMSE_users = np.sqrt(np.mean((np.array(predictions) - np.array(truths)) ** 2))
    print("Hybrid RMSE (cold users):", RMSE_users)
    print("Predictions:", len(predictions))
else:
    print("No predictions for cold users")

# ---- Cold movies ----

predictions_m = []
truths_m = []

for i, row in enumerate(cold_movies_sample.itertuples(index=False)):

    if i % 10000 == 0:
        print("Cold movies processed:", i)

    prediction = predict_hybrid(row.userId, row.movieId)

    if prediction is not None:
        predictions_m.append(prediction)
        truths_m.append(row.rating)

if len(predictions_m) > 0:
    RMSE_movies = np.sqrt(np.mean((np.array(predictions_m) - np.array(truths_m)) ** 2))
    print("Hybrid RMSE (cold movies):", RMSE_movies)
    print("Predictions:", len(predictions_m))
else:
    print("No predictions for cold movies")

print("\nEvaluation complete")