import os
from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the preprocessed recipe data
recipe_data = pd.read_csv('final_repices_all.csv')

seg1=recipe_data[(recipe_data.minutes<=40)&(recipe_data.calories<=400)]
seg2=recipe_data[(recipe_data.minutes<=40)&(recipe_data.calories>400)]
seg3=recipe_data[(recipe_data.minutes>40)&(recipe_data.calories<=400)]
seg4=recipe_data[(recipe_data.minutes>40)&(recipe_data.calories>400)]

seg1.set_index('name', inplace=True)
seg2.set_index('name', inplace=True)
seg3.set_index('name', inplace=True)
seg4.set_index('name', inplace=True)


# Load the pre-trained TF-IDF vectorizers and matrices for each segment
seg1_vectorizer = joblib.load('seg1_vectorizer.pkl')
seg1_tfidf_matrix = joblib.load('seg1_tfidf_matrix.pkl')
seg2_vectorizer = joblib.load('seg2_vectorizer.pkl')
seg2_tfidf_matrix = joblib.load('seg2_tfidf_matrix.pkl')
seg3_vectorizer = joblib.load('seg3_vectorizer.pkl')
seg3_tfidf_matrix = joblib.load('seg3_tfidf_matrix.pkl')
seg4_vectorizer = joblib.load('seg4_vectorizer.pkl')
seg4_tfidf_matrix = joblib.load('seg4_tfidf_matrix.pkl')

# Define a function to get recipe recommendations
def get_recipe_recommendations(user_input, segment):
    if segment == 'SEG1':
        vectorizer = seg1_vectorizer
        tfidf_matrix = seg1_tfidf_matrix
        data = seg1
    elif segment == 'SEG2':
        vectorizer = seg2_vectorizer
        tfidf_matrix = seg2_tfidf_matrix
        data = seg2
    elif segment == 'SEG3':
        vectorizer = seg3_vectorizer
        tfidf_matrix = seg3_tfidf_matrix
        data = seg3
    elif segment == 'SEG4':
        vectorizer = seg4_vectorizer
        tfidf_matrix = seg4_tfidf_matrix
        data = seg4

    user_input_tfidf = vectorizer.transform([user_input])
    cosine_sim = cosine_similarity(user_input_tfidf, tfidf_matrix)
    similar_top5_indices = list(cosine_sim.argsort()[0][-5:])
    recommendations = list(data.iloc[similar_top5_indices].index)
    return recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        segment = request.form['segment']
        recommendations = get_recipe_recommendations(user_input, segment)
        return render_template('index.html', user_input=user_input, segment=segment, recommendations=recommendations)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
