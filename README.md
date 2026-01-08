# Steam Game Recommendation System with TF-IDF

A machine learning-powered game recommendation engine using TF-IDF and cosine similarity, built with Streamlit.

## Features

- **TF-IDF Based Recommendations**: Uses text vectorization to find similar games based on descriptions, tags, and metadata
- **Fast Search**: Offline ML model with pre-computed vectors for instant recommendations
- **Interactive UI**: Built with Streamlit for an intuitive user experience
- **Similarity Filtering**: Adjust similarity threshold to control recommendation quality
- **34,600+ Games**: Recommendations from a comprehensive Steam dataset

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gothari921/steam-game-recccomendation-ml-tfidf.git
cd steam-game-recccomendation-ml-tfidf
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Try the Live App

**[Click here to use the app online!](https://steam-game-recccomendation-ml-tfidf-w6fmxqrdssd4prz2tpehdw.streamlit.app)**

No installation needed - just click and start searching for game recommendations!

## Usage

### Local Installation

Run the Streamlit app on your machine:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### How to Use

1. Enter a game name (e.g., "Dark Souls", "Witcher", "Call of Duty")
2. Click "Search" or press Enter
3. Adjust the number of recommendations (5-20) and similarity threshold in the sidebar
4. View similar games with developer, publisher, and tags information

## Project Structure

- **app.py** - Streamlit web UI application
- **save_model.py** - Script to generate and save ML model components
- **model_vectorizer.pkl** - Pre-trained TF-IDF vectorizer
- **model_vectors.pkl** - Pre-computed feature vectors (13 MB)
- **model_indices.pkl** - Game name to index mapping
- **model_df.pkl** - Game metadata (name, tags, developer, publisher)
- **steam_games.csv** - Original dataset (67 MB)
- **requirements.txt** - Python dependencies

## Model Details

The recommendation system uses:
- **TF-IDF Vectorization**: Converts game descriptions, tags, developer, and publisher info into feature vectors
- **Cosine Similarity**: Calculates similarity between games based on their feature vectors
- **Top-N Recommendation**: Returns the N most similar games to the query game

### Model Parameters
- Max features: 15,000
- N-gram range: (1, 2) - unigrams and bigrams
- Stop words: English
- Min document frequency: 3

## Dataset

The model is trained on 34,633 Steam games with the following features:
- Game description (desc_snippet)
- Release date
- Developer
- Publisher
- Popular tags

## Performance

- Model loading: < 1 second (from pickle files)
- Search and recommendation: < 500ms
- No internet connection required (fully offline)

## Deployment

To deploy on Streamlit Cloud:

1. Push code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and select this repository
4. Streamlit Cloud will automatically install dependencies from `requirements.txt`

## Future Enhancements

- [ ] User rating/feedback system
- [ ] Collaborative filtering recommendations
- [ ] Real-time model updates
- [ ] Game price and reviews integration
- [ ] Download history tracking

## License

MIT License - feel free to use and modify

## Author

Harish Gottimukkula

## Acknowledgments

- Steam dataset source
- Scikit-learn for ML utilities
- Streamlit for the web framework
