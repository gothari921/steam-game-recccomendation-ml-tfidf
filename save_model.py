import pandas as pd
import numpy as np
import joblib
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

# Save all components using joblib (more robust than pickle for numpy versions)
joblib.dump(vectorizer, "model_vectorizer.joblib", compress=3)
joblib.dump(feature_vectors, "model_vectors.joblib", compress=3)
joblib.dump(indices, "model_indices.joblib", compress=3)

# Save dataframe (only relevant columns)
df_minimal = df[["name", "popular_tags", "developer", "publisher"]]
joblib.dump(df_minimal, "model_df.joblib", compress=3)

print("✅ Model saved successfully!")
print("Files created: model_vectorizer.joblib, model_vectors.joblib, model_indices.joblib, model_df.joblib")
