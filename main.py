# %%
##### Global Variables ######
create_new_vectors = False
data_size = 1500

# %%
###### Import Libraries ######
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

import spacy
nlp = spacy.load("en_core_web_md")

import pickle

from tensorflow import keras 
from scipy import sparse

import os

print("Imports complete")

# %%
####### Data Preprocessing Functions #######
def is_not_br(token):
    return ((not ('<br' in token.text)) & (not ('/><br' in token.text)) & (not ('/>' in token.text)) & (token.text != 'br') & (not ('<' in token.text)))
def tokenize(text):
    clean_tokens = []
    for token in nlp(text):
        if (not token.is_stop) & (token.lemma_ != '-PRON-') & (not token.is_punct) & is_not_br(token) & (not token.text.isdigit()):
            clean_tokens.append(token.lemma_.lower())
    return clean_tokens
# Maps 'positive' to 1 and 'negative' to 0
def convert_y(sentiments):
    converted_list = []
    for sentiment in sentiments:
        converted_list.append(1 if sentiment == 'positive' else 0)
    return np.array(converted_list)
# Normalization function before neural network: normalizes all values between 0 and 1
def normalize(arr):
    max_val = np.max(arr)
    min_val = np.min(arr)
    # normalized_array = np.array([2 * ((val - min_val)/(max_val - min_val)) for val in arr])
    normalized_array = 2 * ((arr - min_val)/(max_val - min_val)) - 1
    return normalized_array

#%%
###### Loading Data ######
imdb = pd.read_csv('./imdbDataset.csv')
imdb = imdb.iloc[:data_size]
pd.set_option("display.max_colwidth", None)
print(f"Length : {len(imdb)}")

print("Data Loading Complete")

#%%
##### Word -> Number Encoding ######
##### Bag Of Words
##### Hyperparams: max_features
def bag_of_words(X, y):

    def load_model():
        print("Loading vectorizer...")
        return pickle.load(open("./preencoded_embeddings/tfidf_vectors.pickle", "rb"))
    def create_model(X_data):
        print("Creating vectorizer...")
        bow_transformer = CountVectorizer(analyzer=tokenize, max_features=2000).fit(X_data)
        X_data = bow_transformer.transform(X_data)
        pickle.dump(X_data, open("./preencoded_embeddings/bow_transformer_vectors.pickle", "wb"))
        return X_data

    print("Starting Bag of Words Model")
    if create_new_vectors:
        if os.path.isfile("./preencoded_embeddings/bow_transformer_vectors.pickle"):
            user_input = input("Found vectorizer. Are you sure you still want to make a new vectorizer? (Y/N)")
            if user_input == "N":
                X = load_model()
            elif user_input == "Y":
                os.remove("./preencoded_embeddings/word2vec_reviews.pickle")
                create_model(X)
            else:
                print("Input not in format specified")
        else:
            print("Did not find vectorizer.")
            X = create_model(X)
    else:
        print("Vectorizer Found.")
        X = load_model()
    
    y = convert_y(y)
    print("Bag of Words Model Completed")
    return sparse.lil_matrix(X).toarray(), y

##### TF-IDF Vectorization
##### Hyperparams: max_features
def tf_idf(X, y):

    def load_model():
        print("Loading vectorizer...")
        return pickle.load(open("./preencoded_embeddings/tfidf_vectors.pickle", "rb"))
    def create_model(X_data):
        print("Creating vectorizer...")
        tfidf_transformer = TfidfVectorizer(analyzer=tokenize, max_features=2000).fit(X_data)
        X_data = tfidf_transformer.transform(X_data)
        pickle.dump(X_data, open("./preencoded_embeddings/tfidf_vectors.pickle", "wb"))
        return X_data

    print("Starting TF-IDF Model")
    if create_new_vectors:
        if os.path.isfile("./preencoded_embeddings/bow_transformer_vectors.pickle"):
            user_input = input("Found vectorizer. Are you sure you still want to make a new vectorizer? (Y/N)")
            if user_input == "N":
                X = load_model()
            elif user_input == "Y":
                os.remove("./preencoded_embeddings/word2vec_reviews.pickle")
                create_model(X)
            else:
                print("Input not in format specified")
        else:
            print("Did not find vectorizer.")
            X = create_model(X)
    else:
        print("Vectorizer Found.")
        X = load_model()

    y = convert_y(y)
    print("TF-IDF Model Created")
    return sparse.lil_matrix(X).toarray(), y

##### Pre-trained Word Embeddings (Word2Vec Twitter Model)
##### Hyperparams: 
def word2vec(X, y):

    def load_model():
        print("Loading word vectors...")
        return pickle.load(open("./preencoded_embeddings/word2vec_reviews.pickle", "rb"))
    def create_model(X):
        vectorized_reviews = []
        for index, review in enumerate(X):
            avg_vector = np.zeros(shape=(300))
            for token in tokenize(review):
                avg_vector += nlp(token).vector
            avg_vector = avg_vector / len(X)
            vectorized_reviews.append(avg_vector)
            print(f"Review {index}: Done")
        pickle.dump(vectorized_reviews, open("./preencoded_embeddings/word2vec_reviews.pickle", "wb"))
        return vectorized_reviews

    print("Starting Word2Vec Model")
    if create_new_vectors:
        if os.path.isfile("./preencoded_embeddings/word2vec_reviews.pickle"):
            user_input = input("Found vectorizer. Are you sure you still want to make a new vectorizer? (Y/N)")
            if user_input == "N":
                X = load_model()
            elif user_input == "Y":
                os.remove("./preencoded_embeddings/word2vec_reviews.pickle")
                create_model(X)
            else:
                print("Input not in format specified")
        else:
            print("Did not find vectorizer")
            X = create_model(X)
    else:
        X = load_model()
    y = convert_y(y)    
    print("Word2Vec Model Created")
    return normalize(np.array(X)), y

##### BERT Language Model

##### Model Types ######
##### Logistic Regression
##### Hyperparams: train_test_split, regularization_type, C (inverse of regularization strength), 
def logistic_regression(X, y):
    print("Starting creation of Logistic Regression Model")
    logistic_model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)
    print("Logistic Regression Model created")
    return y_test, y_pred

##### K-Nearest-Neighbours Classifier
##### Hyperparams: n_neighbours, train_test_split
def knn_classifier(X, y, number_neighbors):
    print("Started creation of KNN Classifier")
    knn_classifier = KNeighborsClassifier(n_neighbors=number_neighbors)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    print("KNN Classifier created")
    return y_test, y_pred

##### Naive Bayes Classifier
##### Hyperparams: train_test_split
def naive_bayes(X, y):
    print("Started creation of Naive Bayes Classifier")
    nb_model = MultinomialNB()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    print("Naive Bayes Classifier created")
    return y_test, y_pred

##### Neural Network
##### Hyperparams: train_test_split, Architecture, Optimizer: Learning Rate, Epochs, Batch Size, Initial Weights, Initial Biases
#####              Epochs, Loss Function, Regularization
def neural_network(X, y, architecture_id):
    print("Started creation of Neural Network")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    if architecture_id == 1:
        model = keras.Sequential([
                keras.layers.Dense(300, activation="tanh", kernel_initializer=keras.initializers.RandomUniform(minval=-1, maxval=1), bias_initializer=keras.initializers.TruncatedNormal(mean=0, stddev=0.5)),
                keras.layers.Dense(150, activation="sigmoid", kernel_initializer=keras.initializers.RandomUniform(minval=-1, maxval=1), bias_initializer=keras.initializers.TruncatedNormal(mean=0, stddev=0.5)),
                keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=50, batch_size=64)
    elif architecture_id == 2:
        model = keras.Sequential([
            keras.layers.Dense(2000, activation="relu"),
            keras.layers.Dense(200, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=5, batch_size=64)
    
    model.summary()
    y_pred = model.predict(X_test)
    argmax_predictions = np.array([round(array[0]) for array in y_pred])
    print("Neural Network Created")
    return y_test, argmax_predictions

# %%
##### Applying Models And Printing Accuracy #####

X, y = word2vec(imdb['review'], imdb['sentiment'])
y_test, y_pred = neural_network(X, y, 1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy: ", round(accuracy, 2), ", Precision: ", round(precision, 2), ", Recall: ", round(recall, 2))

##### KNN Classifier Implementation
# max_accuracy = -1
# max_accuracy_neighbors = 0

# for i in range(1, 10):
#     print(f"Number of neighbors: {i}")
#     y_test, y_pred = knn_classifier(X, y, i)

#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)

#     print("Accuracy: ", round(accuracy, 2), ", Precision: ", round(precision, 2), ", Recall ", round(recall, 2))
#     print("\n")

#     if (accuracy > max_accuracy):
#         max_accuracy = accuracy
#         max_accuracy_neighbors = i

# print(f"Maximum Accuracy obtained was {max_accuracy} with {max_accuracy_neighbors}")