import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
df = pd.read_csv("steam_games.csv")

# Select features
selected_features = ["desc_snippet", "release_date", "developer", "publisher", "popular_tags", "name"]

# Fill missing values
for feature in selected_features:
    df[feature] = df[feature].fillna("")

# Combine features
combined_features = (
    df["desc_snippet"] + " " + 
    df["release_date"] + " " + 
    df["developer"] + " " + 
    df["publisher"] + " " + 
    df["popular_tags"] + " " + 
    df["name"]
)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=15000,
    ngram_range=(1, 2),
    min_df=3
)

# Fit and transform
feature_vectors = vectorizer.fit_transform(combined_features)

# Create indices mapping
indices = pd.Series(df.index, index=df['name']).drop_duplicates()

# Save all components
with open("model_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model_vectors.pkl", "wb") as f:
    pickle.dump(feature_vectors, f)

with open("model_indices.pkl", "wb") as f:
    pickle.dump(indices, f)

# Save dataframe (only relevant columns)
df_minimal = df[["name", "popular_tags", "developer", "publisher"]]
with open("model_df.pkl", "wb") as f:
    pickle.dump(df_minimal, f)

print("✅ Model saved successfully!")
print("Files created: model_vectorizer.pkl, model_vectors.pkl, model_indices.pkl, model_df.pkl")
