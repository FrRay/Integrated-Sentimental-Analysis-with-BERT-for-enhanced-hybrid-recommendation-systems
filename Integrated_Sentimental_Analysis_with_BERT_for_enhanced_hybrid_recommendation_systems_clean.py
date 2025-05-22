# 挂载Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# 读取CSV文件
import pandas as pd
# file_path = "/content/drive/My Drive/yelp_training_set_flattened.csv"
file_path = "yelp_training_set_flattened.csv" # TODO
df = pd.read_csv(file_path)
df.head()

# 数据预处理部分
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import string

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocessing(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    cleaned_tokens = []
    for word, tag in pos_tags:
        if word.lower() not in stop_words and word not in string.punctuation:
            wordnet_pos = get_wordnet_pos(tag)
            lemmatized_word = lemmatizer.lemmatize(word.lower(), wordnet_pos)
            cleaned_tokens.append(lemmatized_word)
    return " ".join(cleaned_tokens)

df_sampled = df[:2500]
df_sampled['text'] = df_sampled['text'].apply(preprocessing)
df_sampled['text'].head()

# BERT情感评分部分
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def get_sentiment_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    sentiment_score = torch.argmax(scores).item()
    return sentiment_score

df_sampled['sentimental score'] = df_sampled['text'].apply(get_sentiment_score)
df_sampled['sentimental score'].describe()
df_sampled['sentimental score']
df.info()

# 分类业务类型
restaurant_keywords = ['Restaurant']
hotel_keywords = ['Hotel']

df_sampled['Type'] = df_sampled['business_categories'].apply(
    lambda x: 'Restaurant' if any(word in str(x) for word in restaurant_keywords)
    else ('Hotel' if any(word in str(x) for word in hotel_keywords) else 'Other')
)

restaurant_df = df_sampled[df_sampled['Type'] == 'Restaurant'].reset_index(drop=True)
hotel_df = df_sampled[df_sampled['Type'] == 'Hotel'].reset_index(drop=True)

# 深度矩阵分解预测
import numpy as np
import tensorflow as tf
from tensorflow import keras

def train_deep_mf(df, embedding_dim=8, epochs=20, batch_size=50, validation_split=0.1):
    user_ids = df['review_id'].astype('category').cat.codes.values
    item_ids = df['business_id'].astype('category').cat.codes.values
    ratings = df['stars'].values
    
    num_users = len(np.unique(user_ids))
    num_items = len(np.unique(item_ids))
    
    user_input = keras.layers.Input(shape=(1,))
    item_input = keras.layers.Input(shape=(1,))
    user_embedding = keras.layers.Embedding(num_users, embedding_dim, input_length=1)(user_input)
    item_embedding = keras.layers.Embedding(num_items, embedding_dim, input_length=1)(item_input)
    
    user_vec = keras.layers.Flatten()(user_embedding)
    item_vec = keras.layers.Flatten()(item_embedding)
    concat = keras.layers.Concatenate()([user_vec, item_vec])
    dense1 = keras.layers.Dense(32, activation='relu')(concat)
    dense2 = keras.layers.Dense(16, activation='relu')(dense1)
    output = keras.layers.Dense(1)(dense2)
    
    model = keras.models.Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    
    model.fit([user_ids, item_ids], ratings, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    
    df['DeepMF_predictions'] = model.predict([user_ids, item_ids]).flatten()
    
    return df

train_deep_mf(restaurant_df)
train_deep_mf(hotel_df)
restaurant_df['DeepMF_predictions']
hotel_df['DeepMF_predictions']

# 聚类分析
# !pip install gensim

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler

def kmeans_clustering(df, text_column='text', sentiment_column='sentimental score',
                      vector_size=100, window=5, min_count=1, workers=4,
                      pca_components=20, num_clusters=5):
    
    texts = df[text_column].astype(str).apply(lambda x: x.split())
    
    w2v_model = Word2Vec(sentences=texts, vector_size=vector_size, window=window, sg=1,
                         min_count=min_count, workers=workers)
    
    def text_to_vector(tokens):
        vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)
    
    df['text_vector'] = df[text_column].apply(lambda x: text_to_vector(x.split()))
    
    text_feature_matrix = np.vstack(df['text_vector'])
    text_feature_df = pd.DataFrame(text_feature_matrix, columns=[f'w2v_{i}' for i in range(vector_size)])
    
    numerical_features = df[[sentiment_column]]
    scaler = StandardScaler()
    numerical_features_scaled = scaler.fit_transform(numerical_features)
    numerical_df = pd.DataFrame(numerical_features_scaled, columns=[sentiment_column])
    
    final_features = pd.concat([text_feature_df, numerical_df], axis=1)
    
    pca = PCA(n_components=pca_components)
    final_features_pca = pca.fit_transform(final_features)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(final_features_pca)
    
    return df

kmeans_clustering(restaurant_df)
kmeans_clustering(hotel_df)
restaurant_df['cluster']
hotel_df['cluster']

# 合并特征
featuresh = pd.concat([hotel_df['cluster'], hotel_df['DeepMF_predictions'], hotel_df['sentimental score']], axis=1)
featuresr = pd.concat([restaurant_df['cluster'], restaurant_df['DeepMF_predictions'], restaurant_df['sentimental score']], axis=1)
labelsh = hotel_df['stars']
labelsr = restaurant_df['stars']

# 非负矩阵分解
import numpy as np
from sklearn.decomposition import NMF

def nmf(features, n_components=3, init='random', random_state=42):
    nmf = NMF(n_components=n_components, init=init, random_state=random_state)
    nmf_features = nmf.fit_transform(features)
    hybrid_features = np.hstack((features, nmf_features))
    return hybrid_features

hybrid_faetures_res = nmf(featuresr)
hybrid_faetures_hot = nmf(featuresh)

# 决策树回归
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def decision_tree_regressor(features, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    
    regressor = DecisionTreeRegressor(random_state=random_state)
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'Root Mean Squared Error (RMSE): {rmse}')

print("Decision Tree Regressor for Restaurants:")
DTR = decision_tree_regressor(hybrid_faetures_res, labelsr)
print("Decision Tree Regressor for Hotels:")
DTH = decision_tree_regressor(hybrid_faetures_hot, labelsh)

# 数据描述统计
df.describe()