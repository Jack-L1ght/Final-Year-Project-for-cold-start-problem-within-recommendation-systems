# ---- Imports the required libraries ----

import tkinter as tk
from tkinter import messagebox
import threading

from Recommendation_Engine import recommend_movies, create_user_from_genres

# ---- List of Genres ----

genres = [
    "Action","Adventure","Animation","Children","Comedy",
    "Crime","Documentary","Drama","Fantasy","Film-Noir",
    "Horror","Musical","Mystery","Romance","Sci-Fi",
    "Thriller","War","Western"
]

# ---- Create Users ----
next_user_id = 200949  # start high to avoid clashes


def create_user():

    global next_user_id

    selected = [g for g, var in genre_vars.items() if var.get() == 1]

    if len(selected) == 0:
        messagebox.showerror("Error", "Select at least one genre")
        return

    user_id = next_user_id
    next_user_id += 1

    create_user_from_genres(user_id, selected)

    # update UI
    user_entry.delete(0, tk.END)
    user_entry.insert(0, str(user_id))

    results_box.delete(0, tk.END)
    results_box.insert(tk.END, f"User created successfully (ID: {user_id})")

# ---- Generate recommendations ----


def get_recommendations():
    thread = threading.Thread(target=run_recommendations)
    thread.start()


def run_recommendations():

    try:
        user_id = int(user_entry.get())
    except:
        messagebox.showerror("Error", "Please enter a valid User ID")
        return

    results_box.delete(0, tk.END)
    results_box.insert(tk.END, "Generating recommendations...")

    results = recommend_movies(user_id, top_n=10)

    results_box.delete(0, tk.END)

    if len(results) == 0:
        results_box.insert(tk.END, "No recommendations found")
        return

    for i, movie in enumerate(results, start=1):
        text = f"{i}. {movie['title']}  ⭐ {movie['predicted_rating']}"
        results_box.insert(tk.END, text)

# ---- GUI setup ----


root = tk.Tk()
root.title("Movie Recommendation System")
root.geometry("600x650")

tk.Label(root, text="Movie Recommendation System", font=("Arial", 18)).pack(pady=10)

# ---- User ID input ----

tk.Label(root, text="User ID:").pack()
user_entry = tk.Entry(root)
user_entry.pack(pady=5)

# ---- Selection of Genres ----

tk.Label(root, text="Select Favourite Genres").pack(pady=10)

genre_vars = {}

for i in genres:
    var = tk.IntVar()
    cb = tk.Checkbutton(root, text=i, variable=var)
    cb.pack(anchor="w")
    genre_vars[i] = var

# ---- GUI buttons for creating users and getting the recommendations ----

tk.Button(root, text="Create User", command=create_user).pack(pady=5)

tk.Button(root, text="Get Recommendations", command=get_recommendations).pack(pady=5)

# ---- Prints the results of the recommendations to the text box below the buttons

results_box = tk.Listbox(root, width=80, height=15)
results_box.pack(pady=10)

# ---- Runs the GUI ----

root.mainloop()
