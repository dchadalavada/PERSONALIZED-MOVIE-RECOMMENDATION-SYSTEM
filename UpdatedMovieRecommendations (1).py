import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Reading the movies data into pandas dataframes
df1 = pd.read_csv('./tmdb_5000_credits.csv')
df2 = pd.read_csv('./tmdb_5000_movies.csv')

# Displaying the information about the datasets
print("Credits Dataset Info")
print(df1.head(5))
print()
print("Credits Dataset Columns Information")
print(df1.columns)
print()
print("Movies Dataset Info")
print(df2.head(5))
print()
print("Movies Dataset Columns Information")
print(df2.columns)
print()
print()

# Merging both dataframes on id to have a combined dataset
df1.columns = ['id', 'tittle', 'cast', 'crew']
df2 = df2.merge(df1, on='id')
print(df2.head(5))

# Demographic Filtering

# Mean of votes average
C = df2['vote_average'].mean()

# Converting vote count to quantile
m = df2['vote_count'].quantile(0.9)

print(f"Quantile: {m}")

# Filtering the movies
q_movies = df2.copy().loc[df2['vote_count'] >= m]
print(q_movies.shape)

# Weighted rating function
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

# Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

# Print the top 10 movies
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))

# Calculate the Popular movies
pop = df2.sort_values('popularity', ascending=False)

plt.figure(figsize=(12,4))
plt.barh(pop['title'].head(6), pop['popularity'].head(6), align='center', color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
plt.show()

# Content-based filtering

# Create a TF-IDF(Term Frequency - Inverse Document Frequency ) Vectorizer Object with English stop words removed 
tfidf = TfidfVectorizer(stop_words='english')

# Fill NaN values in 'overview' column with an empty string
df2['overview'] = df2['overview'].fillna('')

# Fit and transform data to construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(df2['overview'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Find the index of the movie title
    idx = indices[title]

    # Calculate pairwise similarity scores of all movies with the given movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies based on similarity scores
    sim_scores.sort(key=lambda x: x[1], reverse=True)

    # Extract indices of the 10 most similar movies
    movie_indices = [i[0] for i in sim_scores[1:11]]

    # Return titles of the top 10 most similar movies
    return df2.iloc[movie_indices]['title']

print("Getting recommendations for 'The Dark Knight Rises' using Cosine Similarity based on plot description: ")
print(get_recommendations('The Dark Knight Rises'))

# Interactive section for user to get recommendations based on movie title starting with a specific letter
while True:
    # Prompt the user to enter an alphabet letter or 'Exit' to terminate the program
    user_input = input("\nEnter an alphabet (A-Z) to list movies, or type 'Exit' to exit: ")
    if user_input.lower() == 'exit':  # Check if the user wants to exit the program
        print("Exiting the program. Thank you for using the movie recommender.")
        break  # Exit the loop and end the program
    if not user_input.isalpha() or len(user_input) != 1:  # Validate user input as a single alphabet character
        print("Invalid input. Please enter a single alphabet letter (A-Z).")
        continue  # Skip the rest of the loop and prompt for input again
    
    # Create a regex pattern to filter movies starting with the specified letter, case-insensitive
    pattern = f'^{user_input}'
    filtered_movies = df2[df2['title'].str.match(pattern, case=False, na=False)]
    if filtered_movies.empty:  # Check if there are any movies that start with the entered letter
        print(f"No movies found starting with '{user_input}'. Try a different letter.")
        continue  # Skip the rest of the loop and prompt for input again

    # Display the list of movies starting with the entered letter
    print("\nMovies starting with", user_input.upper(), ":")
    print(filtered_movies['title'].to_string(index=False))
    
    # Prompt the user to type an exact movie name from the list for recommendations
    movie_title = input("\nType the exact name of the movie from the list above for recommendations, or type 'Back' to choose another letter: ")
    if movie_title.lower() == 'back':  # Allow the user to go back and choose another letter
        continue  # Return to the start of the loop
    # Ensure the entered movie title matches exactly with one in the list
    while movie_title not in filtered_movies['title'].values:
        movie_title = input("Movie not found. Please type an exact name from the list, or type 'Back' to choose another letter: ")
        if movie_title.lower() == 'back':  # Allow the user to cancel and choose another letter
            break  # Exit the while loop
    else:  # If the movie title is found and the user did not type 'Back'
        print(f"\nGetting recommendations based on the movie: {movie_title}")
        # Call the function to get movie recommendations based on the chosen movie
        recommendations = get_recommendations(movie_title)
        # Display the recommended movies
        print("\nRecommended movies:")
        print(recommendations.to_string(index=False))
