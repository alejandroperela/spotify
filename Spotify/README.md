# Spotify Music Data Analysis

A comprehensive exploratory and machine learning project for analyzing and predicting track popularity on Spotify using various regression models.

## 📊 Project Overview

This project analyzes Spotify track data to predict song popularity using multiple machine learning algorithms. The analysis includes data exploration, preprocessing, feature engineering, model training, and comprehensive performance evaluation.

## 📁 Dataset

The datasets have been obtained from the Spotify Global Music Dataset (2009–2025) dataset from Kaggle (https://www.kaggle.com/datasets/wardabilal/spotify-global-music-dataset-20092025/code?datasetId=8708707&sortBy=voteCount)

The project uses two datasets:

### 1. `track_data_final.csv` (Original Dataset)
- **Size**: 8,778 tracks
- **Features**: 15 columns including track metadata, artist information, and album details
- Contains track duration in milliseconds and raw artist genres in list format

### 2. `spotify_data_clean.csv` (Cleaned Dataset)
- **Size**: 8,582 tracks (196 duplicates/invalid entries removed)
- **Features**: 15 columns with preprocessed data
- Track duration converted to minutes
- Artist genres cleaned and standardized

### Dataset Features

| Feature                | Description                              |
|------------------------|------------------------------------------|
| `track_id`             | Unique identifier for each track         |
| `track_name`           | Name of the track                        |
| `track_number`         | Position in the album                    |
| `track_popularity`     | Target variable (0-100 popularity score) |
| `track_duration_min`   | Duration of track in minutes             |
| `explicit`             | Boolean indicating explicit content      |
| `artist_name`          | Name of the artist                       |
| `artist_popularity`    | Artist popularity score (0-100)          |
| `artist_followers`     | Number of followers for the artist       |
| `artist_genres`        | Comma-separated list of genres           |
| `album_id`             | Unique identifier for the album          |
| `album_name`           | Name of the album                        |
| `album_release_date`   | Release date of the album                |
| `album_total_tracks`   | Total number of tracks in the album      |
| `album_type`           | Type (album, single, compilation)        |

## 🔧 Technologies & Libraries

```python
# Data Processing
- pandas
- numpy

# Visualization
- matplotlib
- seaborn

# Machine Learning
- scikit-learn
  - LogisticRegression
  - Ridge Regression
  - Lasso Regression
  - RandomForestRegressor
  - HistGradientBoostingRegressor

# Model Persistence
- joblib
```

## 🚀 Project Workflow

### 1. Data Import & Exploration
- Load both original and clean datasets
- Examine data structure and statistics
- Display sample records and column information

### 2. Data Cleaning & Preprocessing
- Handle missing values
- Remove duplicates
- Convert data types (track duration to minutes)
- Clean and standardize artist genres
- Analyze data distributions

### 3. Feature Engineering
- **Temporal Features**:
  - `release_year`: Extracted from album release date
  - `release_month`: Month of release
  - `release_day`: Day of release
  - `days_since_release`: Days elapsed since release

- **Categorical Encoding**:
  - One-hot encoding for album type
  - Label encoding for artist names and album names

- **Numerical Scaling**:
  - Standard scaling for all numerical features

### 4. Exploratory Data Analysis (EDA)
Comprehensive visualizations including:
- Distribution of track popularity
- Artist popularity vs. track popularity
- Correlation heatmap
- Track duration analysis
- Album type distribution
- Release year trends
- Top artists and genres analysis

### 5. Model Training & Evaluation

Four regression models are trained and compared:

| Model                           | Description                                            |
|---------------------------------|--------------------------------------------------------|
| **Random Forest Regressor**     | Ensemble method with 100 estimators, max depth 15      |
| **Histogram Gradient Boosting** | Gradient boosting with learning rate 0.1, max depth 10 |
| **Ridge Regression**            | L2 regularization with alpha=10.0                      |
| **Lasso Regression**            | L1 regularization with alpha=1.0                       |

### 6. Performance Metrics

Each model is evaluated using:
- **RMSE (Root Mean Square Error)**: Measures prediction accuracy
- **R² Score**: Coefficient of determination
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **Training vs. Test Performance**: Identifies overfitting

### 7. Visualizations

The analysis generates comprehensive visualizations:
1. **Model Comparison - RMSE**: Bar chart comparing test RMSE across models
2. **Model Comparison - R²**: Bar chart comparing R² scores
3. **Actual vs. Predicted**: Scatter plot showing prediction accuracy
4. **Residual Plot**: Identifies prediction patterns and biases
5. **Feature Importance**: Top 10 most important features (for tree-based models)
6. **Error Distribution**: Histogram of prediction errors

Output saved as: `track_popularity_analysis.png`

## 📈 Results

The best performing model is automatically selected based on the highest Test R² score. The analysis provides:

- Complete performance metrics for all models
- Feature importance rankings
- Visual analysis of predictions
- Residual analysis for model validation

## 🔍 Key Insights

The analysis reveals important factors affecting track popularity:
- Artist popularity and follower count are strong predictors
- Temporal features (release date, days since release) impact popularity
- Album type and genre influence track success
- Track characteristics like duration and explicit content play a role

## 💾 Model Persistence

The best performing model can be saved for future predictions:

```python
import joblib
joblib.dump(best_model, 'track_popularity_model.pkl')

# Load and use
loaded_model = joblib.load('track_popularity_model.pkl')
predictions = loaded_model.predict(X_new)
```

## 📝 Usage

### Clone the repo
```bash
git clone https://github.com/username/project.git
cd ~/Documents/Projects/Personal-Projects/Spotify
```

### Set up the virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Prerequisites
```bash
pip install -r requirements.txt
```

### Launch Visual Studio Code
```bash
code .
```

## 📂 Project Structure

```
.
├── data/
│   ├── track_data_final.csv            # Original dataset
│   └── spotify_data_clean.csv          # Cleaned dataset
├── images/
│   └── track_popularity_analysis.png   # Output visualizations
├── notebooks/
│   └── spotify_music_data.ipynb        # Main analysis notebook
├── requirements.txt                    # Library requirements
└── README.md                           # This file
```

## 🎯 Future Improvements

- Add more advanced feature engineering (interaction terms, polynomial features)
- Implement cross-validation for more robust model evaluation
- Try deep learning models (Neural Networks)
- Incorporate audio features (tempo, energy, danceability) if available
- Develop a web interface for real-time predictions
- Time series analysis for popularity trends

## 📊 Data Quality Notes

- 196 records were removed during cleaning (duplicates or invalid entries)
- Missing values in artist genres were handled appropriately
- Outliers were retained to preserve data distribution
- Data spans multiple years and genres for diverse representation

## 🤝 Contributing

Contributions are welcome! Areas for contribution:
- Additional feature engineering
- Alternative modeling approaches
- Enhanced visualizations
- Performance optimization
- Documentation improvements

## 📄 License

This project is available for educational and research purposes.

## 👨‍💻 Author

Name: Alejandro Perela
Email: alejandro.perela.posada@gmail.com
LinkedIn: https://www.linkedin.com/in/alejandro-perela-posada-575408a7/

**Note**: This analysis is based on Spotify data and is intended for educational purposes. The models' predictions should be interpreted as analytical insights rather than definitive popularity forecasts.
