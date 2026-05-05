# ------ imports the surprise library ------

from surprise import Dataset, Reader, SVD
import pandas as pd
import os
import joblib
import numpy as np

# Dataset: This is used to build the datasets for surprise
# Reader: This is used to define the rating scale
# SVD: A matrix factorisation model
# Accuracy: Used to generate the RMSE ( Root Mean squared Error)

# ------ Create folders if they do not exist ------

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ------ imports the filtered dataset from the CSV files generated in Filtering_Movie_Lens ------

training_data = pd.read_csv(
    "Filtered_Data/training_data.csv",
    usecols=["userId", "movieId", "rating"]
)

cold_users_test = pd.read_csv(
    "Filtered_Data/cold_users_test.csv",
    usecols=["userId", "movieId", "rating"]
)

cold_movie_test = pd.read_csv(
    "Filtered_Data/cold_movie_test.csv",
    usecols=["userId", "movieId", "rating"]
)

train_df = training_data
cold_users_df = cold_users_test
cold_movies_df = cold_movie_test

# ------ Builds the surprise dataset ------
# This will define the range of rating within the dataset

reader = Reader(rating_scale=(0.5, 5.0))

# ------ Changes the pandas training dataframe into a surprise dataset ------

train_data = Dataset.load_from_df(
    train_df,
    reader
)

# ------ Generates and trains the factorisation model (SVD)------
# The following model is trained only using "warm" data
train_set = train_data.build_full_trainset()

algo = SVD(
    n_factors=50,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

print("Training SVD model...")
algo.fit(train_set)

print("Saving trained model...")
joblib.dump(algo, "models/svd_baseline.pkl")

# ------ Prepares the cold user/movies test set for the evaluation------
# Uses the same format as for the warm data which is; userID, MovieId and rating

def compute_rmse_batched(df, batch_size=50000):

    squared_error = 0
    count = 0

    for start in range(0, len(df), batch_size):

        batch = df.iloc[start:start + batch_size]

        for row in batch.itertuples(index=False):

            pred = algo.predict(row.userId, row.movieId).est
            error = pred - row.rating

            squared_error += error ** 2
            count += 1

    rmse = np.sqrt(squared_error / count)

    return rmse, count


# ------ Evaluates the training model on the cold user test set ------

print("\nEvaluating cold users...")

rmse_cold_users, size_users = compute_rmse_batched(cold_users_df)

# ------ Evaluates cold movies ------

print("Evaluating cold movies...")

rmse_cold_movies, size_movies = compute_rmse_batched(cold_movies_df)

# ------ Prints the summaries ------


print("\nBaseline MF results")
print("-------------------")
print("Cold users RMSE :", rmse_cold_users)
print("Cold movies RMSE:", rmse_cold_movies)

print("Cold user predictions:", size_users)
print("Cold movie predictions:", size_movies)

# ------ Save results for report / reproducibility ------


results = pd.DataFrame([{
    "cold_users_rmse": rmse_cold_users,
    "cold_movies_rmse": rmse_cold_movies,
    "cold_users_size": size_users,
    "cold_movies_size": size_movies
}])

results.to_csv(
    "results/baseline_svd_results.csv",
    index=False
)

print("\nResults saved to results/baseline_svd_results.csv")

