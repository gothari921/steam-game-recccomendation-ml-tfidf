# 🎮 Steam Game Recommendation System - Getting Started

## Project Overview

A machine learning-powered recommendation engine that suggests Steam games based on similarity analysis using **TF-IDF** vectorization and **Cosine Similarity**.

---

## 🌐 External Resources

### Dataset
- **Kaggle Dataset**: [Steam Games Dataset](https://www.kaggle.com/datasets/nikdavis/steam-store-games)
- **Alternative Source**: [Steam Dataset on Kaggle](https://www.kaggle.com/search?q=steam+games)
- **Dataset Size**: 34,633 games with descriptions, tags, developer, and publisher information

### Steam Platform
- **Official Steam Store**: [https://store.steampowered.com](https://store.steampowered.com)
- **Steam Community**: [https://steamcommunity.com](https://steamcommunity.com)
- **Steam API Documentation**: [https://developer.valvesoftware.com/wiki/Steam_Web_API](https://developer.valvesoftware.com/wiki/Steam_Web_API)

### Project Links
- **GitHub Repository**: [steam-game-recccomendation-ml-tfidf](https://github.com/gothari921/steam-game-recccomendation-ml-tfidf)
- **Live Demo**: [Streamlit Cloud App](https://steam-game-recccomendation-ml-tfidf-w6fmxqrdssd4prz2tpehdw.streamlit.app)

---

## 📊 Project Structure

```
steam-game-recccomendation-ml-tfidf/
├── app.py                      # Streamlit web application
├── save_model.py              # Model training and saving script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── GET_INFO.md               # This file
├── steam_games.csv           # Dataset (67 MB)
├── games_reccomendation.ipynb # Original Jupyter notebook
├── model_vectorizer.joblib   # TF-IDF vectorizer model
├── model_vectors.joblib      # Pre-computed feature vectors
├── model_indices.joblib      # Game name to index mapping
└── model_df.joblib           # Game metadata
```

---

## 🚀 Quick Start Guide

### 1. **Clone the Repository**
```bash
git clone https://github.com/gothari921/steam-game-recccomendation-ml-tfidf.git
cd steam-game-recccomendation-ml-tfidf
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Run the App**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 💻 Features

✅ **Fast Recommendations** - Sub-500ms response time  
✅ **34,600+ Games** - Comprehensive Steam database  
✅ **Offline Processing** - No internet connection required  
✅ **Adjustable Results** - Control number of recommendations (5-20)  
✅ **Similarity Filtering** - Filter results by similarity threshold  
✅ **Rich Metadata** - See developer, publisher, and game tags  

---

## 📈 How It Works

### Algorithm Pipeline
```
Game Query
    ↓
TF-IDF Vectorization (convert text to numerical features)
    ↓
Cosine Similarity Calculation (compare with all games)
    ↓
Top-N Selection (get most similar games)
    ↓
Display Results (with metadata)
```

### Model Configuration
| Parameter | Value |
|-----------|-------|
| Max Features | 15,000 |
| N-gram Range | (1, 2) - unigrams and bigrams |
| Stop Words | English |
| Min Document Frequency | 3 |
| Similarity Metric | Cosine Similarity |

---

## 🎯 Example Searches

Try these games to see recommendations:

- **Dark Souls** → Similar action RPGs
- **The Witcher** → Story-driven fantasy games
- **Call of Duty** → Multiplayer FPS games
- **Assassin's Creed** → Adventure action games
- **Ghost of Tsushima** → Samurai/historical games

---

## 📦 Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.13** | Programming language |
| **Streamlit** | Web UI framework |
| **Scikit-learn** | ML library (TF-IDF, Cosine Similarity) |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Joblib** | Model serialization |

---

## 📥 Dataset Information

### Data Source
- **Source**: Steam Store Games Dataset (Kaggle)
- **Size**: 67 MB (34,633 games)
- **Format**: CSV

### Columns Used
| Column | Description |
|--------|-------------|
| `name` | Game title |
| `desc_snippet` | Game description excerpt |
| `release_date` | Release date |
| `developer` | Development studio |
| `publisher` | Publishing company |
| `popular_tags` | User-assigned tags |

### Downloading the Dataset

**Option 1: Download from Kaggle**
1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets/nikdavis/steam-store-games)
2. Click "Download" button
3. Place `steam_games.csv` in project folder

**Option 2: Use Kaggle API**
```bash
kaggle datasets download -d nikdavis/steam-store-games
unzip steam-store-games.zip
```

---

## 🔧 Configuration

### Customize Recommendations
Edit `app.py` to adjust:

```python
# Default number of recommendations
top_n = 10

# Minimum similarity threshold
similarity_threshold = 0.1

# TF-IDF parameters in save_model.py
max_features = 15000
ngram_range = (1, 2)
min_df = 3
```

---

## 📊 Performance Metrics

- **Model Training Time**: ~2 minutes
- **Model Size**: ~18 MB (compressed with joblib)
- **Prediction Time**: < 500ms per query
- **Memory Usage**: ~500 MB (vectors + metadata)
- **Database Coverage**: 34,633 games

---

## 🐛 Troubleshooting

### Issue: "No module named 'sklearn'"
**Solution**: Run `pip install -r requirements.txt`

### Issue: "Error loading models"
**Solution**: Ensure `.joblib` files exist in project folder

### Issue: "Game not found"
**Solution**: Try searching with partial game name or different spelling

### Issue: "Streamlit app won't start"
**Solution**: Check Python version (3.8+) and ports (8501 available)

---

## 🚀 Deployment

### Deploy on Streamlit Cloud
1. Push code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select repository and `app.py`
5. Deploy automatically!

### Deploy on Other Platforms
- **Heroku**: Use Procfile + requirements.txt
- **AWS**: Deploy as Lambda + API Gateway
- **Docker**: Create Dockerfile with dependencies

---

## 📝 Model Training

To retrain the model with new data:

```bash
python save_model.py
```

This will:
1. Load `steam_games.csv`
2. Preprocess text data
3. Train TF-IDF vectorizer
4. Compute feature vectors
5. Save model components as `.joblib` files

---

## 📚 Additional Resources

### Machine Learning Concepts
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Content-Based Recommendation Systems](https://developers.google.com/machine-learning/recommendation/content-based/basics)

### Related Projects
- [Movie Recommendation System](https://github.com/topics/recommendation-system)
- [Game Similarity Analysis](https://github.com/topics/game-similarity)
- [Text-Based Recommendations](https://github.com/topics/tfidf)

---

## 📞 Support & Contact

- **GitHub Issues**: [Report bugs](https://github.com/gothari921/steam-game-recccomendation-ml-tfidf/issues)
- **Kaggle Dataset Issues**: [Steam Games Dataset](https://www.kaggle.com/datasets/nikdavis/steam-store-games)
- **Streamlit Support**: [Streamlit Docs](https://docs.streamlit.io)

---

## 📄 License

MIT License - Free to use and modify for personal and commercial projects.

---

## 🙏 Acknowledgments

- **Dataset**: Niklas Davis (Kaggle)
- **Steamworks API**: Valve Corporation
- **ML Libraries**: Scikit-learn team
- **Web Framework**: Streamlit team

---

## 🎮 Next Steps

1. ✅ Clone the repository
2. ✅ Install dependencies
3. ✅ Download dataset from Kaggle (if not included)
4. ✅ Run `python save_model.py` to train model (if needed)
5. ✅ Launch app with `streamlit run app.py`
6. ✅ Search for your favorite game and get recommendations!

---

**Last Updated**: January 8, 2026  
**Version**: 1.0.0  
**Status**: Active & Maintained ✨

