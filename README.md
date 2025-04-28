# Integrated Sentiment Analysis with BERT for Enhanced Hybrid Recommendation Systems

This repository contains implementation code for sentiment analysis using BERT within a hybrid recommendation system framework. The code demonstrates how sentiment analysis can be combined with collaborative filtering techniques to improve recommendation quality for different business types.

## Project Overview

This project implements a hybrid recommendation system that combines:

- BERT-based sentiment analysis
- Deep Matrix Factorization
- Word embedding clustering
- Non-negative Matrix Factorization (NMF)
- Decision Tree Regression

The system processes Yelp reviews to generate recommendations for restaurants and hotels, considering both user ratings and sentiment scores derived from review text.

## Dataset

The implementation uses the Yelp dataset (`yelp_training_set_flattened.csv`), containing business reviews with the following key fields:

- `review_id`: Unique identifier for each review
- `business_id`: Identifier for the business being reviewed
- `text`: Review text content
- `stars`: User rating (1-5 stars)
- `business_categories`: Categories describing the business

## Implementation Details

### 1. Data Preprocessing

- Loading and examining the Yelp dataset
- Text preprocessing pipeline including:
  - Tokenization
  - Part-of-speech tagging
  - Stop word removal
  - Lemmatization
- Business type classification (Restaurant/Hotel/Other)

### 2. Sentiment Analysis

- Utilizes BERT model (`nlptown/bert-base-multilingual-uncased-sentiment`) for sentiment classification
- Processes preprocessed review text to generate sentiment scores
- Integration of sentiment scores as features for the recommendation system

### 3. Deep Matrix Factorization

- Neural network-based collaborative filtering
- User and item embedding layers with dimension=8
- Dense layers (32 → 16 → 1) to capture non-linear interactions
- Trained separately for restaurant and hotel data subsets

### 4. Clustering Analysis

- Word2Vec embeddings for review text (vector_size=100)
- Feature combination of text vectors and sentiment scores
- Dimensionality reduction using PCA
- K-means clustering to group similar reviews

### 5. Hybrid Feature Generation

- Non-negative Matrix Factorization (NMF) to create latent features
- Combination of collaborative filtering predictions, sentiment scores, and cluster assignments

### 6. Rating Prediction

- Decision Tree Regressor for final rating prediction
- Performance evaluation using RMSE (Root Mean Squared Error)
- Separate models for restaurant and hotel recommendations

## Requirements

The implementation requires the following libraries:

- pandas
- nltk
- torch
- transformers
- tensorflow
- numpy
- gensim
- scikit-learn

## Usage

The code is designed to be run in Google Colab with the dataset stored in Google Drive. The main workflow is:

1. Mount Google Drive and load the dataset
2. Preprocess review text
3. Extract sentiment scores using BERT
4. Separate data by business type (restaurants/hotels)
5. Train Deep Matrix Factorization models
6. Generate clusters based on text and sentiment
7. Create hybrid features using NMF
8. Train and evaluate Decision Tree Regressors

## Results

The system generates separate recommendation models for restaurants and hotels, with RMSE scores provided for each domain. The integration of sentiment analysis with collaborative filtering demonstrates how natural language understanding can enhance traditional recommendation approaches.
