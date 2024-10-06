import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model


app = Flask(__name__)

# Load the model and the dataset
model = load_model('movie_model.keras')
movies_df = pd.read_csv('movies.csv')

# Convert genres into dummy variables
genres_dummies = movies_df['genres'].str.get_dummies(sep='|')
cosine_sim = cosine_similarity(genres_dummies)

# Define the movie indices (mapping of titles to DataFrame indices)
movie_indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

# Recommend movies based on similar genres
def recommend_movies_nn(title, num_recommendations=5):
    if title not in movie_indices:
        return []

    # Get the index of the movie that matches the title
    idx = movie_indices[title]

    # Get the pairwise similarity scores for all movies with the input movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the most similar movies
    sim_scores = sim_scores[1:num_recommendations + 1]  # Exclude the first one (itself)

    # Get the movie indices and titles
    movie_indices_sim = [i[0] for i in sim_scores]
    recommended_titles = movies_df['title'].iloc[movie_indices_sim]

    return recommended_titles.tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.json.get('title')
    recommendations = recommend_movies_nn(movie_title)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
