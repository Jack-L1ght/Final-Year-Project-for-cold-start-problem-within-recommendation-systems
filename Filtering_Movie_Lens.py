# ------ imports the required libraries ------

import pandas as pd


def filtering():

    # ------ Loading MovieLens dataset files ------

    movie_ratings = pd.read_csv("ml-32m/ratings.csv")
    movies = pd.read_csv("ml-32m/movies.csv")
    movie_tags = pd.read_csv("ml-32m/tags.csv")

    # ------ displays the data sparsity for the dataset ------

    movie_ratings["rating"].describe()
    movie_ratings["rating"].value_counts().sort_index()

    # ------ Sparsity calculation ------

    # counts only unique users, movies and the total ratings

    num_users = movie_ratings["userId"].nunique()
    num_items = movie_ratings["movieId"].nunique()
    num_ratings = len(movie_ratings)

    sparsity = 1 - (num_ratings / (num_users * num_items))
    print(f"Sparsity: {sparsity:.4f}")

    # ------ Missing Values ------

    # Metadata that has missing data can cause harm to the other set of data

    movie_ratings.isnull().sum()
    movies.isnull().sum()
    movie_tags.isnull().sum()

    # ------ Relevant attributes ------

    # Collecting relevant attributes to be used for later use

    rating_features = movie_ratings[["userId", "movieId", "rating"]]
    movie_features = movies[["movieId", "title", "genres"]]
    tag_features = movie_tags[["userId", "movieId", "tag"]]

    # ------ Removes all the duplicate records ------

    # Remove duplicate records to prevent bias in interaction counts

    movie_ratings.drop_duplicates(inplace=True)
    movies.drop_duplicates(inplace=True)
    movie_tags.drop_duplicates(inplace=True)

    # ------ Replace missing genres ------

    movies["genres"] = movies["genres"].fillna("Unknown")

    # ------ Remove invalid ratings ------

    movie_ratings = movie_ratings[(movie_ratings["rating"] >= 0.5) & (movie_ratings["rating"] <= 5.0)]

    # ------ Filtering inactive users and rare movies ------

    # Gets rid of data that can cause inconsistency for the end results

    min_user_ratings = 25 # having it below 25 will not filter any users
    min_movie_ratings = 5

    user_counts = movie_ratings["userId"].value_counts()
    movie_counts = movie_ratings["movieId"].value_counts()

    # ------ creates the users that will be valid after the filtering has been completed ------

    valid_users = user_counts[user_counts >= min_user_ratings].index
    valid_items = movie_counts[movie_counts >= min_movie_ratings].index

    # ------ uses the filtered out users to separate the movies and users that will be used later ------

    filtered_ratings = movie_ratings[
        movie_ratings["userId"].isin(valid_users) &
        movie_ratings["movieId"].isin(valid_items)
        ].copy()

    filtered_ratings_with_movies = filtered_ratings.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
    )

    # ------ Prints the users and movies before the filter of 5 ratings minimal for the movies and the users.------

    print(f"Users before filtering: {movie_ratings['userId'].nunique()}")
    print(f"Movies before filtering: {movie_ratings['movieId'].nunique()}")

    print(f"Users after filtering: {filtered_ratings['userId'].nunique()}")
    print(f"Movies after filtering: {filtered_ratings['movieId'].nunique()}")

    # ------ Will print an example of the csv file in the terminal ------

    # filtered_ratings_with_movies = filtered_ratings_with_movies.drop(columns=["timestamp"])

    # filtered_ratings_with_movies.to_csv(
        # "filtered_ratings_with_movies.csv",
        # index=False)

    # print(filtered_ratings_with_movies[["movieId", "title", "genres"]].head())

    # ------ creating a copy of the filtered csv file and the cleaned dataset ------

    training_data = filtered_ratings_with_movies

    # ------ will gather the cold users ------

    user_counts = movie_ratings["userId"].value_counts()

    cold_users = user_counts[user_counts < 25].index

    cold_users_test = movie_ratings[
        movie_ratings["userId"].isin(cold_users) &
        movie_ratings["movieId"].isin(training_data["movieId"].unique())
    ].copy()

    # ------ will gather the cold movies ------

    movie_counts = movie_ratings["movieId"].value_counts()

    cold_movies = movie_counts[movie_counts < 25].index

    cold_movie_test = movie_ratings[
        movie_ratings["movieId"].isin(cold_movies) &
        movie_ratings["userId"].isin(training_data["userId"].unique())
        ].copy()

# Turns the filtered dataset into csv files to be easily read later on
    training_data.to_csv("Filtered_Data/training_data.csv", index=False)
    cold_users_test.to_csv("Filtered_Data/cold_users_test.csv", index=False)
    cold_movie_test.to_csv("Filtered_Data/cold_movie_test.csv", index=False)
    movies.to_csv(
        "Filtered_Data/movies.csv", index=False)


# This code will run the filteration of the Movielens dataset
rerun_code = input("Want to rerun the filtering code? ")
if rerun_code.lower() == ["yes", "yeah"]:
    filtering()
else:
    print("Code has finished")
    exit()
