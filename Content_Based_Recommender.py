# ------ Imports the required libraries ------

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import joblib
import os

print("Starting the content based model ...")

# ------ Checks if the model folder exists------

# used to make sure the folder exists to be able to store the trained models to be used for later
os.makedirs("models", exist_ok=True)

# ------ Loads the MovieLens Dataset------
print("Loading the MovieLens dataset ...")

# Obtains all the required csv files to be used for the model
training_data = pd.read_csv("Filtered_Data/training_data.csv")
cold_users_test = pd.read_csv("Filtered_Data/cold_users_test.csv")
cold_movie_test = pd.read_csv("Filtered_Data/cold_movie_test.csv")
movies = pd.read_csv("Filtered_Data/movies.csv")

print("MovieLens dataset loaded")

# ------ Filter for relevant movies only------

# Only keeps the movies that were present in the training data only to helpr reduce noise and size
print("Filtering the movies used within the dataset ...")

used_movie_IDs = pd.concat([
    training_data["movieId"],
    cold_users_test["movieId"],
    cold_movie_test["movieId"]
]).unique()

movies_used = movies[movies["movieId"].isin(used_movie_IDs)].copy()

print("Movies remaining to filter:", len(movies_used))

# ------ preprocessing the genre data------

# Changes the genre strings into TF-IDF vectorisation to be suitable for the  lines of code to follow
print("Preparing genre text...")

movies_used["genres_split"] = movies_used["genres"].str.replace("|", " ", regex=False)

# ------ TF-IDF feature extraction------
# Changes the genres into numerical values to represent item content

print("Building TF-IDF vectors...")

tfidf = TfidfVectorizer()

genres_vector = tfidf.fit_transform(movies_used["genres_split"])

# Normalise the vectors for cosine similarity
genres_vector = normalize(genres_vector)

print("TF-IDF matrix built")

# ------ Creates a mapping for movie index------
# Maps the movieID index to the TF-IDF matrix
print("Creating the movie index...")

movieId_index = dict(
    zip(movies_used["movieId"], range(len(movies_used)))
)

print("Movie index size:", len(movieId_index))

# ------ Saves the models for reuse------
# Saves the TF-IDF model,feature matrix and the indexing structure for the GUI system

print("Saving TF-IDF models...")

joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(genres_vector, "models/movie_vectors.pkl")
joblib.dump(movieId_index, "models/movie_index.pkl")

print("TF-IDF models saved")

# ------ Builds the user profiles------
# Each of the users are represented as an average of their weights for each liked movie genre

print("Building user profiles...")

user_profiles = {}

for i, (user_id, group) in enumerate(training_data.groupby("userId")):

    if i % 10000 == 0:
        print("Processed users:", i)

    profile = None
    total_weight = 0.0

    for row in group.itertuples(index=False):

        mid = row.movieId
        rating = row.rating

        if mid not in movieId_index:
            continue

        movie_vec = genres_vector[movieId_index[mid]]
        weighted_vec = movie_vec * rating

        if profile is None:
            profile = weighted_vec
        else:
            profile = profile + weighted_vec

        total_weight += rating

    if profile is not None and total_weight > 0:
        user_profiles[user_id] = normalize(profile / total_weight)

print("Finished building user profiles")
print("Total profiles:", len(user_profiles))

# ------ Saves the profiles of the users------
# Which is used later for the prediction and integration of the hybrid model

print("Saving user profiles...")

joblib.dump(user_profiles, "models/user_profiles.pkl")

print("User profiles saved")

# ------ prediction function------
# Computes the similarities between the user profiles as well as the movie features
# displays the predicted rating scaled to the MovieLens rating range which is up to 5 Stars

print("Creating prediction function...")

def predict_content_rating(user_id, movie_id):

    if user_id not in user_profiles:
        return None

    if movie_id not in movieId_index:
        return None

    user_vec = user_profiles[user_id]
    movie_vec = genres_vector[movieId_index[movie_id]]

    sim = user_vec.dot(movie_vec.T)[0, 0]

    return 0.5 + sim * (5.0 - 0.5)

print("Prediction function ready")

# ------ evaluation of the CBF------
# Takes random sample from the training data to simulate different interactions

print("Evaluating cold users...")

preds = []
truths = []

sample_users = training_data.sample(50000)

for i, row in enumerate(sample_users.itertuples(index=False)):

    if i % 10000 == 0:
        print("Processed:", i)

    prediction = predict_content_rating(row.userId, row.movieId)

    if prediction is not None:
        preds.append(prediction)
        truths.append(row.rating)

print("Number of predictions (cold users):", len(preds))

if len(preds) > 0:
    rmse = np.sqrt(np.mean((np.array(preds) - np.array(truths)) ** 2))
    print("Content-based RMSE (cold users):", rmse)
else:
    print("No predictions generated for cold users.")

# ------ evaluation of the cold movie scenario------
# Tests the model performance on cold items

print("Evaluating cold movies...")

preds_m = []
truths_m = []

for i, row in enumerate(cold_movie_test.itertuples(index=False)):

    if i % 100000 == 0:
        print("Cold movie predictions processed:", i)

    prediction = predict_content_rating(row.userId, row.movieId)

    if prediction is not None:
        preds_m.append(prediction)
        truths_m.append(row.rating)

print("Number of predictions (cold movies):", len(preds_m))

if len(preds_m) > 0:
    rmse_m = np.sqrt(np.mean((np.array(preds_m) - np.array(truths_m)) ** 2))
    print("Content-based RMSE (cold movies):", rmse_m)
else:
    print("No predictions generated for cold movies.")

print("Content-based evaluation complete")